from typing import Union, List, Tuple
from pathlib import Path

import polars as pl


# TODO: Allow chaining method: This may require huge amount of refactoring
# TODO: Categorical columns sometimes needs pl.StringCache when comparing strings
# Currently, elements are compared by string, since it is not that slow for now.
class NiaDataPathExtractor:
    def __init__(
            self,
            dataset_dir: str="/datasets/nia/",
            pattern: str=(
                r"(?P<split>[^/]+)/"
                r"(?P<type>[^/]+)/"
                r"(?P<collector>[^/]+)/"
                r".*?"
                r"(?P<channel>[^/]+)/"
                r"(?P<filename>[^/]+)$"
            ),
        ) -> None:
        # Directory should have a trailing slash
        if not dataset_dir.endswith("/"):
            dataset_dir += "/"
        self.dataset_dir = dataset_dir
        self.pattern = pattern

        self.paths = pl.Series("path", map(str, Path(dataset_dir).rglob("*.*")))

        # TODO: match, pair, filter, drop_nulls have to be run on the user-side
        self.matched_df = self.match(dataset_dir + pattern)
        self.paired_df = self.pair()
    
    def match(self, pattern: Union[str, None]=None) -> pl.DataFrame:
        if pattern is None:
            pattern = self.dataset_dir + self.pattern
        
        # Get all file paths and match subdirectory names
        matched_df = (
            self.paths
            .str.extract_groups(pattern)
            .struct.unnest()
            .with_columns(self.paths)
        )

        # Get stem and extension from filename
        matched_df = matched_df.with_columns(
            pl.col("filename").str.split(".").alias(".splits"),
        ).with_columns(
            pl.col(".splits").list.first().alias("stem"),
            pl.col(".splits").list.last().alias("extension"),
        ).drop(".splits")

        # Get features from stem
        timeslot_values = ["mrh", "day", "lunch", "afterschool", "erh", "night"]
        weather_values = ["clear", "rainy", "foggy"]

        matched_df = matched_df.with_columns(
            pl.col("stem").str.extract(r"([A-Z]+)", 1).alias("sensor"),
            pl.col("stem").str.extract_all(r"[A-Z]\d{2}").list.to_struct(
                fields=["scene", "road"],
            ).alias("scenario_codes"),
            pl.col("stem").str.extract(f"({'|'.join(timeslot_values)})").alias("timeslot"),
            pl.col("stem").str.extract(f"({'|'.join(weather_values)})").alias("weather"),
            pl.col("stem").str.extract(r"(\d{8})").alias("annotation_id")
        ).unnest("scenario_codes")

        # TODO: match(), pair() have to be run on the user-side
        self.matched_df = matched_df
        return matched_df
    
    # TODO: pair method should be refactored to another class or external function
    def pair(self) -> pl.DataFrame:
        collections_df = self.matched_df.filter(
            pl.col("extension").is_in(["pcd", "png"]),
            pl.col("split").is_in(["1.Training", "2.Validation", "3.Test"]),
            pl.col("type") == "1.원천데이터",
        ).with_columns(
            pl.col("split").replace({
                "1.Training": "train",
                "2.Validation": "valid",
                "3.Test": "test",
            }),
        ).rename({"path": "collection_path"})

        annotations_df = self.matched_df.filter(
            pl.col("extension") == "json",
            pl.col("split").is_in(["1.Training", "2.Validation", "3.Test", "6.서브라벨링"]),
            pl.col("type") == "2.라벨링데이터",
        ).rename({"path": "annotation_path"}).drop("split", "collector")

        paired_df = annotations_df.join(
            collections_df.select(pl.col("split", "stem", "collector", "collection_path")),
            on="stem",
        ).select([
            "split", "stem",
            "collector", "channel", "sensor", "scene", "road", "timeslot", "weather", "annotation_id",
            "collection_path", "annotation_path",
        ]).sort("stem")

        # TODO: match, pair, filter, drop_nulls have to be run on the user-side
        self.paired_df = paired_df
        return paired_df


class DataFrameSplitter:
    def __init__(
            self,
            groups: List[str]=["collector", "scene", "road", "timeslot", "weather"],
            splits: List[str]=["train", "valid", "test"],
            ratios: List[float]=[8, 1, 1],
            seed: Union[int, None]=231111,
        ) -> None:

        self.groups = groups
        self.splits = splits
        self.ratios = ratios
        self.seed = seed

    # TODO: This breaks the original ordering. Returned DataFrame is automatically sorted.
    def split(self, df: pl.DataFrame, random: bool=False) -> pl.DataFrame:
        groups = self.groups
        seed = self.seed

        df_with_split = (
            df
            .group_by([*groups, "annotation_id"], maintain_order=True)  # Group by collections gathered together
            .all()  # Zip groups where each group has 6 collections from 6 channels
            .select("stem", "collector", "channel",
                    "sensor", "scene", "road", "timeslot", "weather", "annotation_id",
                    "collection_path", "annotation_path")  # Reorder columns
            .with_row_count("index")  # Add group index
            .group_by(groups, maintain_order=True)  # Group by same scenarios
            .map_groups(lambda x: x.with_columns(
                self.assign_split(len(x), random=random, seed=seed + x.get_column("index").min()),
            ))  # Each Scenario will be splited into [train, valid, test] sets with ratio [8, 1, 1]
            .explode("stem", "channel", "sensor", "collection_path", "annotation_path")  # unzip groups
        )
        return df_with_split
    
    def random_split(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.split(df, random=True)
    
    def assign_split(
            self,
            num_samples: int,
            random: bool=False,
            seed: int=231111,
        ) -> pl.Series:
        splits = self.splits
        ratios = self.ratios
            
        split_info_df = pl.DataFrame([
            pl.Series("split", splits),
            pl.Series("ratio", ratios),
        ]).with_columns(
            (pl.col("ratio") / pl.col("ratio").sum()).alias("proportion")
        ).with_columns(
            (pl.col("ratio").cum_sum() / pl.col("ratio").sum() * num_samples).round().cast(pl.Int64).alias("end_index"),
        ).with_columns(
            pl.col("end_index").shift(1, fill_value=0).alias("start_index"),
        ).with_columns(
            (pl.col("end_index") - pl.col("start_index")).alias("length")
        ).select("split", "ratio", "proportion", "start_index", "end_index", "length")

        split_series = split_info_df.with_columns(
            pl.int_ranges("start_index", "end_index"),
        ).explode("int_range").drop_nulls().get_column("split")

        if random is True:
            split_series = split_series.shuffle(seed)

        return split_series


class NiaDataPathProvider:
    def __init__(
            self,
            reader: NiaDataPathExtractor,
            splitter: Union[DataFrameSplitter, None]=None,
            exclude_filenames: List[str]=[],
        ) -> None:
        self.reader = reader
        self.splitter = splitter
        self.exclude_filenames = exclude_filenames
        
        if self.splitter:
            self.nia_df = self.splitter.random_split(self.reader.paired_df)
        else:
            self.nia_df = self.reader.paired_df
    
    def get_split_data_list(
            self,
            channels: Union[str, List[str]],
            splits: Union[str, List[str]],
        ) -> List[Tuple[str, str]]:

        df = self.nia_df
        df = self.exclude_files(df)

        if type(channels) is str:
            channels = [channels]
        
        if type(splits) is str:
            splits = [splits]

        return df.filter(
            pl.col("channel").is_in(channels),
            pl.col("split").is_in(splits),
        ).select(
            "collection_path",
            "annotation_path",
        ).rows()

    def exclude_files(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            ~pl.col("collection_path").str.extract(r"[^/]+$", 0).is_in(self.exclude_filenames),
            ~pl.col("annotation_path").str.extract(r"[^/]+$", 0).is_in(self.exclude_filenames),
        )
