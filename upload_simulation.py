#!/usr/bin/env python3
import asyncio
import aiohttp.client_exceptions
import dataclasses
import ujson as json
import re
import time
from asyncio import Semaphore
from pathlib import Path
from typing import Union, Sequence, Type

import aiohttp
import click
import msgpack

from movici_simulation_core.data_tracker.data_format import extract_dataset_data


class Client:
    def __init__(
        self,
        host,
        mimetype="application/json",
        session_cls: Type[aiohttp.ClientSession] = aiohttp.ClientSession,
        max_concurrent=10,
    ):
        self.session_cls = session_cls
        self.host = host
        self.headers = {"Accept": mimetype}
        self.concurrent_requests = Semaphore(max_concurrent)

    async def login(self, username, password):

        resp = await self.request(
            "POST",
            self.auth + "/user/login",
            json={"username": username, "password": password},
        )
        self.headers["Authorization"] = resp["session"]

    async def request(self, method, uri, **kwargs) -> Union[dict, bytes]:
        async with self.concurrent_requests:
            t0 = now()
            url = self.host + uri
            resp = await retry(
                lambda: self.session.request(
                    method=method, url=url, headers=self.headers, **kwargs
                ),
                on=aiohttp.client_exceptions.ServerTimeoutError,
                n_tries=3,
            )
            if "application/json" in resp.content_type:
                out = await resp.json()
            elif "application/x-msgpack" in resp.content_type:
                out = msgpack.unpackb(await resp.read(), raw=False)
            else:
                out = await resp.read()
            click.echo(f"{url} [{now() - t0:.3f}] s")

            return out

    async def get(self, uri):
        return await self.request("GET", uri)

    async def post(self, uri, **kwargs):
        return await self.request("POST", uri, **kwargs)

    @property
    def data_engine(self):
        return "/data-engine/v4"

    @property
    def auth(self):
        return "/auth/v1"

    async def get_projects(self):
        return await self.get(f"{self.data_engine}/projects")

    async def post_update(self, scenario_uuid, payload):
        return await self.post(
            f"{self.data_engine}/scenarios/{scenario_uuid}/updates", json=payload
        )

    async def delete_timeline(self, scenario_uuid):
        return await self.request(
            "DELETE", f"{self.data_engine}/scenarios/{scenario_uuid}/timeline"
        )

    async def create_timeline(self, scenario_uuid):
        return await self.request("POST", f"{self.data_engine}/scenarios/{scenario_uuid}/timeline")

    async def get_dataset(self, dataset_uuid):
        return await self.get(f"{self.data_engine}/datasets/{dataset_uuid}/data")

    async def get_datasets(self, project_uuid):
        return await self.get(f"{self.data_engine}/projects/{project_uuid}/datasets")

    async def get_scenarios(self, project_uuid):
        return await self.get(f"{self.data_engine}/projects/{project_uuid}/scenarios")

    async def get_scenario(self, scenario_uuid):
        return await self.get(f"{self.data_engine}/scenarios/{scenario_uuid}")

    async def __aenter__(self):
        self.session = get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()


async def retry(coro_func, *, on=Exception, n_tries=-1):
    err = None
    while n_tries != 0:
        try:
            return await coro_func()
        except on as e:
            err = e
            n_tries -= 1

    if err:
        raise err


@dataclasses.dataclass
class UpdateFile:
    dataset: str
    timestamp: int
    iteration: int
    path: Path

    def __post_init__(self):
        self.timestamp = int(self.timestamp)
        self.iteration = int(self.iteration)


class ScenarioUploader:
    client: Client
    mimetypes = {"json": "application/json", "msgpack": "application/x-msgpack"}

    def __init__(
        self,
        host,
        update_dir: Path,
        project_name=None,
        scenario_name=None,
        mimetype="json",
        client_cls: Type[Client] = Client,
        concurrent=1,
    ):
        self.host = host
        self.update_dir = Path(update_dir)
        self.client_cls = client_cls
        self.project_name = project_name
        self.scenario_name = scenario_name
        self.mimetype = self.mimetypes[mimetype]
        self.concurrent = concurrent

    async def upload_simulation(self, username, password):
        async with self.client_cls(
            self.host, mimetype=self.mimetype, max_concurrent=self.concurrent
        ) as client:
            self.client = client
            await self.client.login(username, password)
            project_uuid = await self.get_project_uuid()
            scenario_uuid = await self.get_scenario_uuid(project_uuid)
            scenario = await self.get_scenario(scenario_uuid)
            if scenario.get("has_timeline"):
                await self.delete_timeline(scenario_uuid)
            await self.create_timeline(scenario_uuid)
            dataset_uuid_map = {
                dataset["name"]: dataset["uuid"] for dataset in scenario["datasets"]
            }
            update_files = self.iter_update_files_sorted(self.update_dir)
            tasks = tuple(
                self.process_update(scenario_uuid, file, dataset_uuid_map) for file in update_files
            )
            t0 = now()
            await asyncio.gather(*tasks)

            click.echo(f"processed {len(tasks)} files in {now() - t0:.3f} s")

    async def process_update(self, scenario_uuid, update: UpdateFile, dataset_uuid_map: dict):
        payload = self.parse_update(update, dataset_uuid_map)
        return await self.client.post_update(scenario_uuid, payload)

    @classmethod
    def iter_update_files_sorted(cls, updates_dir: Path):
        files = sorted(
            cls.iter_update_files(updates_dir), key=lambda u: (u.timestamp, u.iteration)
        )
        it = 0
        for file in files:
            file.iteration = it
            yield file
            it += 1

    @staticmethod
    def iter_update_files(updates_dir: Path):
        matcher = re.compile(r"t(?P<timestamp>\d+)_(?P<iteration>\d+)_(?P<dataset>\w+)\.json")
        for file in updates_dir.glob("*.json"):
            if not (match := matcher.match(file.name)):
                continue
            values = match.groupdict()
            yield UpdateFile(path=file, **values)

    @staticmethod
    def parse_update(file: UpdateFile, dataset_uuid_mapping: dict) -> dict:
        upd = json.loads(file.path.read_text())
        try:
            ((name, data),) = extract_dataset_data(upd)
        except ValueError as e:
            raise ValueError(f"update {file.path} has multiple datasets") from e
        return {
            "name": file.dataset,
            "dataset_uuid": dataset_uuid_mapping[file.dataset],
            "timestamp": file.timestamp,
            "iteration": file.iteration,
            "model_name": "data_collector",
            "model_type": "data_collector",
            "data": data,
        }

    async def get_scenario(self, scenario_uuid):
        return await self.client.get_scenario(scenario_uuid)

    async def get_scenario_uuid(self, project_uuid):
        scenario_name = self.scenario_name
        resp = await self.client.get_scenarios(project_uuid)
        scenarios = {scenario["name"]: scenario["uuid"] for scenario in resp["scenarios"]}
        if scenario_name is None:
            scenario_name = prompt_choices(list(scenarios))
        scenario_uuid = scenarios[scenario_name]
        return scenario_uuid

    async def get_project_uuid(self):
        project_name = self.project_name
        resp = await self.client.get_projects()
        projects = {project["name"]: project["uuid"] for project in resp["projects"]}
        if project_name is None:
            project_name = prompt_choices(list(projects))
        project_uuid = projects[project_name]
        return project_uuid

    async def delete_timeline(self, scenario_uuid):
        return await self.client.delete_timeline(scenario_uuid)

    async def create_timeline(self, scenario_uuid):
        return await self.client.create_timeline(scenario_uuid)

    async def initialize_client(self):
        self.client.session = get_session()
        return self


def get_session() -> aiohttp.ClientSession:
    timeout = aiohttp.ClientTimeout(sock_connect=20, sock_read=50 * 60, total=None)
    return aiohttp.ClientSession(raise_for_status=True, timeout=timeout)


def now():
    return time.time()


def prompt_choices(choices: Sequence[str]):
    """
    Display a prompt for the given choices

    e.g.
        1) a
        2) b
        3) c
        1-3: ?

    Args:
        choices: the choices for the user to choose from
    """
    sorted_choices = sorted(choices)
    for i, option in enumerate(sorted_choices):
        click.echo("{number}) {choice}".format(number=i + 1, choice=option))
    value = click.prompt("1-{}".format(len(choices)), type=int) - 1
    if value < 0 or value >= len(choices):
        raise click.ClickException("Invalid choice.")
    return sorted_choices[value]


@click.command()
@click.option("-h", "--host", help="Simulation Engine Host", default="https://platform.movici.nl")
@click.option("-p", "--project", help="Project name", default=None)
@click.option("-s", "--scenario", help="scenario name", default=None)
@click.option("-d", "--sim_dir", help="Simulation directory")
@click.option("-i", "--init_data_dir", help="Init data directory", default=None)
@click.option("-U", "--username", prompt=True, help="username")
@click.option("-P", "--password", prompt=True, hide_input=True)
@click.option("-c", "--concurrent", help="Number of concurrent requests", default=5)
def main(host, username, password, scenario, project, sim_dir, init_data_dir, concurrent):
    """Get a full (predefined scenario) from the simulation engine"""
    downloader = ScenarioUploader(
        host=host,
        project_name=project,
        scenario_name=scenario,
        concurrent=concurrent,
        update_dir=sim_dir,
    )
    asyncio.run(downloader.upload_simulation(username=username, password=password))


if __name__ == "__main__":
    main()
