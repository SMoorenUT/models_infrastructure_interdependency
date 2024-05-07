# This is the original file from Sahand unedited

import geopandas
import pandas
import numpy as np
from dataset_converters.base import BaseDatasetConverter
from dataset_converters.exceptions import DatasetEmptyError
from movici.dataprocessing import Entity, DataSetV3MetaData
from dataset_converters.properties import (
    Geometry_Linestring2d,
    Geometry_X,
    Geometry_Y,
    Id,
    DisplayName,
    Reference,
    Shape_Length,
    Topology_FromNodeId,
    Topology_ToNodeId,
    Transport_MaxSpeed,
    Transport_Capacity_Hours,
    Transport_PassengerDemand,
    Transport_DomesticCargoDemand,
    Transport_InternationalCargoDemand,
    Transport_PassengerFlow,
    Transport_Layout,
    Transport_CargoFlow,
    Labels
)


class RoadSegmentEntity(Entity):
    entity_name = "road_segment_entities"
    id = Id()
    reference = Reference()
    display_name = DisplayName()
    from_node_id = Topology_FromNodeId()
    to_node_id = Topology_ToNodeId()
    linestring_2d = Geometry_Linestring2d()
    layout = Transport_Layout()
    capacity = Transport_Capacity_Hours()
    length = Shape_Length()
    max_speed = Transport_MaxSpeed()
    passengers_flow = Transport_PassengerFlow(special=-1)
    cargo_flow = Transport_CargoFlow(special=-1)


class VirtualLinkEntity(Entity):
    entity_name = "virtual_link_entities"
    id = Id()
    from_node_id = Topology_FromNodeId()
    to_node_id = Topology_ToNodeId()
    linestring_2d = Geometry_Linestring2d()


class TransportNodeEntity(Entity):
    entity_name = "transport_node_entities"
    x = Geometry_X()
    y = Geometry_Y()
    reference = Reference()
    id = Id()


class VirtualNodeEntity(Entity):
    entity_name = "virtual_node_entities"
    id = Id()
    x = Geometry_X()
    y = Geometry_Y()
    display_name = DisplayName()
    reference = Reference()
    passengers_demands = Transport_PassengerDemand()
    domestic_cargo_demands = Transport_DomesticCargoDemand()
    international_cargo_demands = Transport_InternationalCargoDemand()
    labels = Labels()


class RitriRoadConverter(BaseDatasetConverter):
    def __init__(
        self, dataset_name: str, display_name: str, postfix: str = "", epsg: int = 28992, **_
    ):
        super().__init__(
            dataset_name=dataset_name, display_name=display_name, postfix=postfix, epsg=epsg
        )
        self._geo_data_segments = None
        self._geo_data_nodes = None
        self._geo_data_virtual_nodes = None
        self._geo_data_virtual_links = None
        self._data_passengers_od = None
        self._data_domestic_cargo_od = None
        self._data_international_cargo_od = None

    # noinspection PyMethodOverriding
    def initialize_from_file(
        self,
        road_segments_file: str,
        road_transport_nodes_file: str,
        virtual_nodes_file: str,
        virtual_links_file: str,
        passengers_od_file: str,
        domestic_cargo_od_file: str,
        international_cargo_od_file: str,
        **_,
    ):
        self._geo_data_nodes = geopandas.read_file(road_transport_nodes_file)
        data_roads = geopandas.read_file(road_segments_file)
        self._geo_data_segments = data_roads.sort_values(by=["link_id"]).reset_index(drop=True)
        data_virtual_nodes = geopandas.read_file(virtual_nodes_file)
        self._geo_data_virtual_nodes = data_virtual_nodes.sort_values(
            by=["fid_agg_dn"]
        ).reset_index(drop=True)
        self._geo_data_virtual_links = geopandas.read_file(virtual_links_file)

        self._data_passengers_od = pandas.DataFrame((np.load(passengers_od_file)), dtype="float")
        self._data_domestic_cargo_od = pandas.DataFrame(
            np.load(domestic_cargo_od_file), dtype="float"
        )
        self._data_international_cargo_od = pandas.DataFrame(
            np.load(international_cargo_od_file), dtype="float"
        )

        self._epsg = self._create_epsg_ref(self._geo_data_nodes)
        self._dataset.metadata = self._create_meta_information()

    def initialize_from_postgres(self, connection, schema: str, **_):
        super().initialize_from_postgres(connection=connection, schema=schema)

        self._geo_data_nodes = geopandas.GeoDataFrame.from_postgis(
            "select id, geometry from road_transport_nodes",
            self._db_connection,
            geom_col="geometry",
        )

        self._geo_data_segments = geopandas.GeoDataFrame.from_postgis(
            "select * from road_segments",
            self._db_connection,
            geom_col="geometry",
        )

        self._geo_data_virtual_nodes = geopandas.GeoDataFrame.from_postgis(
            "select *  from road_virtual_nodes",
            self._db_connection,
            geom_col="geometry",
        )

        self._data_passengers_od = pandas.read_sql(
            "select * from road_passengers_od", self._db_connection, coerce_float=True
        )

        self._data_domestic_cargo_od = pandas.read_sql(
            "select * from domestic_cargo_od", self._db_connection, coerce_float=True
        )
        self._data_international_cargo_od = pandas.read_sql(
            "select * from international_cargo_od", self._db_connection, coerce_float=True
        )

        self._geo_data_virtual_links = geopandas.GeoDataFrame.from_postgis(
            "select * from road_virtual_links",
            self._db_connection,
            geom_col="geometry",
        )

        if len(self._geo_data_nodes) == 0 or len(self._geo_data_segments) == 0:
            raise DatasetEmptyError

        if len(self._data_passengers_od) != len(self._data_passengers_od.columns):
            raise ValueError("Passengers OD matrix is not symmetric")

        self._epsg = self._create_epsg_ref(self._geo_data_nodes)
        self._dataset.metadata = self._create_meta_information()

    def _create_meta_information(self) -> DataSetV3MetaData:
        return DataSetV3MetaData(
            name=self._dataset_name,
            type="transport_network",
            display_name=self._display_name,
            epsg_code=self._epsg,
        )

    def convert(self):

        for index, row in self._geo_data_segments.iterrows():
            road_segment = self.create_road_segment_entities(row)
            self._dataset.add_entity(road_segment)

        for index, row in self._geo_data_nodes.iterrows():
            transport_node = self.create_transport_node_entities(row)
            self._dataset.add_entity(transport_node)

        for index, row in self._geo_data_virtual_nodes.iterrows():
            international_cargo_demands = self._data_international_cargo_od.loc[
                row["fid_agg_dn"]
            ].values.tolist()
            domestic_cargo_demands = self._data_domestic_cargo_od.loc[
                row["fid_agg_dn"]
            ].values.tolist()
            passengers_demands = self._data_passengers_od.loc[row["fid_agg_dn"]].values.tolist()
            virtual_node = self.create_virtual_node_entities(
                row, passengers_demands, domestic_cargo_demands, international_cargo_demands
            )
            self._dataset.add_entity(virtual_node)

        for index, row in self._geo_data_virtual_links.iterrows():
            virtual_link = self.create_virtual_link_entities(row)
            self._dataset.add_entity(virtual_link)

        self._dataset.set_link(RoadSegmentEntity.from_node_id, TransportNodeEntity)
        self._dataset.set_link(RoadSegmentEntity.to_node_id, TransportNodeEntity)
        self._dataset.set_link(VirtualLinkEntity.from_node_id, VirtualNodeEntity)
        self._dataset.set_link(VirtualLinkEntity.to_node_id, TransportNodeEntity)
        self._dataset.generate_ids()

    @staticmethod
    def create_transport_node_entities(row):
        transport_node = TransportNodeEntity(
            id=row["id"],
            x=np.round(row["geometry"].x, 3),
            y=np.round(row["geometry"].y, 3),
        )

        transport_node.reference = str(row['id'])
        return transport_node

    @staticmethod
    def create_virtual_node_entities(
        row, passengers_demands, domestic_cargo_demands, international_cargo_demands
    ):
        labels = []
        for label in row["labels"].split(","):
            labels.append(label)

        virtual_node = VirtualNodeEntity(
            x=np.round(row["geometry"].x, 3),
            y=np.round(row["geometry"].y, 3),
            id=row["id"],
            display_name=row["fid_agg_dn"],
            reference=str(row["node_id"]),
            passengers_demands=passengers_demands,
            domestic_cargo_demands=domestic_cargo_demands,
            international_cargo_demands=international_cargo_demands,
            labels=labels,
        )
        return virtual_node

    @staticmethod
    def create_virtual_link_entities(row):

        linestring2d = [
            [np.round(float(i), 3) for i in nested] for nested in list(row["geometry"].coords)
        ]

        virtual_link = VirtualLinkEntity(
            linestring_2d=linestring2d,
            from_node_id=row["source"],
            to_node_id=row["target"],
            id=row["id"],
        )
        return virtual_link

    @staticmethod
    def create_road_segment_entities(row):
        linestring2d = [
            [np.round(float(i), 3) for i in nested] for nested in list(row["geometry"].coords)
        ]

        layout = []
        for i in row["layout"].split(","):
            layout.append(int(i))

        road_segment = RoadSegmentEntity(
            id=row["link_id"],
            display_name=row["display_name"],
            linestring_2d=linestring2d,
            length=np.round(row["length"], 3),
            from_node_id=row["source"],
            to_node_id=row["target"],
            capacity=row["capacity_per_lane"],
            max_speed=round(row["_max_speed"] / 3.6, 3),  # Convert to m/s
            reference=row["link_id"],
            layout=layout
        )
        return road_segment
