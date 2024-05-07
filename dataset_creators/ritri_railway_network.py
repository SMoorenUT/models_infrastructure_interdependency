import geopandas
import pandas
import numpy as np
from dataset_converters.base import BaseDatasetConverter
from dataset_converters.exceptions import DatasetEmptyError
from movici.dataprocessing import Entity, DataSetV3MetaData
from dataset_converters.properties import (
    Connection_ToIds,
    Geometry_Linestring2d,
    Geometry_X,
    Geometry_Y,
    Id,
    DisplayName,
    Reference,
    Shape_Length,
    Topology_FromNodeId,
    Topology_ToNodeId,
    Transport_CargoVehicleMaxSpeed,
    Transport_Capacity_Hours,
    Transport_PassengerDemand,
    Transport_PassengerVehicleMaxSpeed,
    Transport_DomesticCargoDemand,
    Transport_InternationalCargoDemand,
    Transport_Layout,
    Transport_PassengerFlow,
    Transport_CargoFlow,
    Transport_CargoAllowed,
    Labels,
    Transport_PassengerVehicleBaseFrequency,
    Transport_PassengerVehicleFrequency,
)


class TrackSegmentEntity(Entity):
    cargo_vehicle_max_speed = Transport_CargoVehicleMaxSpeed()
    entity_name = "track_segment_entities"
    id = Id()
    reference = Reference()
    layout = Transport_Layout()
    from_node_id = Topology_FromNodeId()
    to_node_id = Topology_ToNodeId()
    linestring_2d = Geometry_Linestring2d()
    capacity = Transport_Capacity_Hours(special=-1)
    length = Shape_Length()
    # max_speed = Transport_MaxSpeed()
    passenger_flow = Transport_PassengerFlow(special=-1)
    passenger_vehicle_max_speed = Transport_PassengerVehicleMaxSpeed()
    cargo_flow = Transport_CargoFlow(special=-1)
    cargo_allowed = Transport_CargoAllowed()


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
    id = Id()


class VirtualNodeEntity(Entity):
    entity_name = "virtual_node_entities"
    id = Id()
    x = Geometry_X()
    y = Geometry_Y()
    display_name = DisplayName()
    reference = Reference()
    passenger_demands = Transport_PassengerDemand()
    domestic_cargo_demands = Transport_DomesticCargoDemand()
    international_cargo_demands = Transport_InternationalCargoDemand()
    passenger_vehicle_frequency = Transport_PassengerVehicleFrequency()
    labels = Labels()


class RitriRailwayNetworkConverter(BaseDatasetConverter):
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
        self._data_passenger_od = None
        self._data_domestic_cargo_od = None
        self._data_international_cargo_od = None
        self._data_passenger_vehicle_frequency_od = None

    # noinspection PyMethodOverriding
    def initialize_from_file(
            self,
            track_segments_file: str,
            transport_nodes_file: str,
            virtual_nodes_file: str,
            virtual_links_file: str,
            passenger_od_file: str,
            domestic_cargo_od_file: str,
            international_cargo_od_file: str,
            passenger_vehicle_frequency_od_file: str,
            **_,
    ):
        self._geo_data_nodes = geopandas.read_file(transport_nodes_file)
        data_tracks = geopandas.read_file(track_segments_file)
        self._geo_data_segments = data_tracks.sort_values(by=["id"]).reset_index(drop=True)
        data_virtual_nodes = geopandas.read_file(virtual_nodes_file)
        self._geo_data_virtual_nodes = data_virtual_nodes.sort_values(
            by=["od_order"]
        ).reset_index(drop=True)
        self._geo_data_virtual_links = geopandas.read_file(virtual_links_file)

        self._data_passenger_od = pandas.DataFrame((np.load(passenger_od_file)), dtype="float")
        self._data_domestic_cargo_od = pandas.DataFrame(np.load(domestic_cargo_od_file), dtype="float")
        self._data_international_cargo_od = pandas.DataFrame(np.load(international_cargo_od_file), dtype="float")
        self._data_passenger_vehicle_frequency_od = pandas.DataFrame(np.load(passenger_vehicle_frequency_od_file),
                                                                     dtype="float")

        self._epsg = self._create_epsg_ref(self._geo_data_nodes)
        self._dataset.metadata = self._create_meta_information()

    def initialize_from_postgres(self, connection, schema: str, **_):
        super().initialize_from_postgres(connection=connection, schema=schema)

        self._geo_data_segments = geopandas.GeoDataFrame.from_postgis(
            "select * track_segments",
            self._db_connection,
            geom_col="geometry",
        )

        self._geo_data_nodes = geopandas.GeoDataFrame.from_postgis(
            "select * from transport_nodes",
            self._db_connection,
            geom_col="geometry",
        )

        self._geo_data_virtual_nodes = geopandas.GeoDataFrame.from_postgis(
            "select * virtual_nodes",
            self._db_connection,
            geom_col="geometry",
        )

        self._geo_data_virtual_links = geopandas.GeoDataFrame.from_postgis(
            "select * virtual_links",
            self._db_connection,
            geom_col="geometry",
        )

        if len(self._geo_data_nodes) == 0 or len(self._geo_data_segments) == 0:
            raise DatasetEmptyError

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
            track_segment = self.create_track_segment_entities(row)
            self._dataset.add_entity(track_segment)

        for index, row in self._geo_data_nodes.iterrows():
            transport_node = self.create_transport_node_entities(row)
            self._dataset.add_entity(transport_node)

        for index, row in self._geo_data_virtual_nodes.iterrows():
            domestic_cargo_demands = self._data_domestic_cargo_od.loc[row["od_order"]].values.tolist()
            international_cargo_demands = self._data_international_cargo_od.loc[row["od_order"]].values.tolist()
            passenger_demands = self._data_passenger_od.loc[row["od_order"]].values.tolist()
            passenger_vehicle_transport_frequency = self._data_passenger_vehicle_frequency_od.loc[
                row["od_order"]].values.tolist()
            virtual_node = self.create_virtual_node_entities(
                row, passenger_demands, domestic_cargo_demands, international_cargo_demands,
                passenger_vehicle_transport_frequency
            )
            self._dataset.add_entity(virtual_node)

        for index, row in self._geo_data_virtual_links.iterrows():
            virtual_link = self.create_virtual_link_entities(row)
            self._dataset.add_entity(virtual_link)

        self._dataset.set_link(TrackSegmentEntity.from_node_id, TransportNodeEntity)
        self._dataset.set_link(TrackSegmentEntity.to_node_id, TransportNodeEntity)
        self._dataset.set_link(VirtualLinkEntity.from_node_id, VirtualNodeEntity)
        self._dataset.set_link(VirtualLinkEntity.to_node_id, TransportNodeEntity)
        self._dataset.generate_ids()

    @staticmethod
    def create_transport_node_entities(row):
        transport_node = TransportNodeEntity(
            id=row["id"],
            x=row["geometry"].x,
            y=row["geometry"].y,
        )
        return transport_node

    @staticmethod
    def create_virtual_node_entities(row, passenger_demands, domestic_cargo_demands,
                                     international_cargo_demands, passenger_vehicle_transport_frequency):
        virtual_node = VirtualNodeEntity(
            x=np.round(row["geometry"].x, 3),
            y=np.round(row["geometry"].y, 3),
            id=row["od_order"],
            reference=row["id"],
            # labels=row["point_abbreviation"],
            passenger_demands=passenger_demands,
            domestic_cargo_demands=domestic_cargo_demands,
            international_cargo_demands=international_cargo_demands,
            passenger_vehicle_frequency=passenger_vehicle_transport_frequency,
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
    def create_track_segment_entities(row):
        linestring2d = [
            [np.round(float(i), 3) for i in nested] for nested in list(row["geometry"].coords)
        ]

        layout = []
        for i in row["layout"].split(","):
            layout.append(int(i))
        track_segment = TrackSegmentEntity(
            id=row["id"],
            reference=row["id"],
            linestring_2d=linestring2d,
            length=np.round(row["length_m"], 3),
            from_node_id=row["source"],
            to_node_id=row["target"],
            passenger_vehicle_max_speed=row["passenger_train_max_speed_m/s"],
            cargo_vehicle_max_speed=row["cargo_train_max_speed_m/s"],
            layout=layout,
            cargo_allowed=row["cargo_route_exists"],
            capacity=row["trajectory_total_capacity(hr)"]
        )

        return track_segment
