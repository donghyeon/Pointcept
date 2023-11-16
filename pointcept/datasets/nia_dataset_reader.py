from pathlib import Path

import polars as pl


# TODO: Allow chaining method: This may require huge amount of refactoring
class NiaDataPathExtractor:
    def __init__(
            self,
            dataset_dir="/datasets/nia/",
            pattern=(
                r"(?P<type>[^/]+)/"
                r"(?P<collector>[^/]+)/"
                r".*?"
                r"(?P<channel>[^/]+)/"
                r"(?P<filename>[^/]+)$"
            )
        ) -> None:
        # Directory should have a trailing slash
        if not dataset_dir.endswith("/"):
            dataset_dir += "/"
        self.dataset_dir = dataset_dir

        # TODO: match, pair, filter, drop_nulls have to be run on the user-side
        self.matched_df = self.match(dataset_dir + pattern)
        self.paired_df = self.pair()
        self.df = self.paired_df.drop_nulls()
    
    @property
    def paths(self) -> pl.Series:
        return pl.Series("path", map(str, Path(self.dataset_dir).rglob("*.*")))
    
    def match(self, pattern: str) -> pl.DataFrame:
        # Get all file paths and match subdirectory names
        paths = self.paths
        matched_df = (
            paths
            .str.extract_groups(pattern)
            .struct.unnest()
            .with_columns(paths)
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
                fields=["code_1", "code_2"],
            ).alias("scenario_codes"),
            pl.col("stem").str.extract(f"({'|'.join(timeslot_values)})").alias("timeslot"),
            pl.col("stem").str.extract(f"({'|'.join(weather_values)})").alias("weather"),
            pl.col("stem").str.extract(r"(\d{8})").alias("annotation_id")
        ).unnest("scenario_codes")
        return matched_df
    
    def pair(self) -> pl.DataFrame:
        # Group by stem and channel
        paired_df = self.matched_df.group_by("stem", "channel").agg(
            pl.all().exclude("filename", "path", "extension").first(),
            pl.col("path").filter(pl.col("extension") != "json").first().name.prefix("collection_"),
            pl.col("path").filter(pl.col("extension") == "json").first().name.prefix("annotation_"),
        )

        # # TODO: Categorical columns sometimes needs pl.StringCache when comparing strings
        # # Currently, elements are compared by string, since it is not that slow for now.
        # # Drop nulls and get all features with complete pairs
        # categorical_columns = [
        #     "channel", "volume", "scene",
        #     "sensor", "code_1", "code_2", "timeslot", "weather",
        # ]
        # paired_df = paired_df.drop_nulls().with_columns(
        #     pl.col(categorical_columns).cast(pl.Categorical).cat.set_ordering("lexical"),
        # )

        return paired_df
    
    def filter_channels(self, channels: list[str]) -> pl.DataFrame:
        return self.df.filter(
            pl.col("channel").is_in(channels),
        )


class DataFrameSplitter:
    def __init__(
            self,
            groups=["channel"],
            splits=["train", "valid", "test"],
            ratios=[8, 1, 1],
            seed=231111,
        ) -> None:

        self.groups = groups
        self.splits = splits
        self.ratios = ratios
        self.seed = seed

    # TODO: Looks ugly. Vectorize operators as much as possible.
    # TODO: This breaks the original ordering. Returned DataFrame is automatically sorted.
    # TODO: Needs "stem" to sort. Check whether sorting by groups[0] gives consistent results.
    def split(self, df: pl.DataFrame, random=False) -> pl.DataFrame:
        groups = self.groups
        splits = self.splits
        ratios = self.ratios
        seed = self.seed

        split_ratio_df = pl.DataFrame([
            pl.Series("split", splits),
            pl.Series("ratio", ratios),
        ]).with_columns(
            (pl.col("ratio") / pl.col("ratio").sum()).alias("proportion")
        )

        split_dfs = []
        for group, grouped_df in df.sort("stem").with_row_count("index").group_by(groups, maintain_order=True):
            num_samples = len(grouped_df)
            if num_samples == 0:
                continue

            slice_indices_df = split_ratio_df.with_columns(
                (pl.col("proportion").cumsum() * num_samples).round().cast(pl.Int64).alias("end_index"),
            ).with_columns(
                pl.col("end_index").shift(1, fill_value=0).alias("start_index"),
            ).with_columns(
                (pl.col("end_index") - pl.col("start_index")).alias("length")
            )

            split_df = grouped_df.with_row_count("group_index").select(
                pl.col("index"),
                pl.col("group_index"),
                pl.lit(None).alias("split"),
            )

            for split, end_index in slice_indices_df.rows_by_key(("split", "end_index")):
                split_df = split_df.with_columns(
                    pl.when((pl.col("group_index") < end_index) & pl.col("split").is_null())
                    .then(pl.lit(split))
                    .otherwise(pl.col("split"))
                    .alias("split")
                )
            
            split_series = split_df.get_column("split")
            if random:
                seed += 1
                split_series = split_series.sample(num_samples, shuffle=True, seed=seed)
            
            split_dfs.append(
                split_df.select(
                    pl.col("index"),
                    split_series.alias("split"),
                ),
            )
        
        split_series = pl.concat(split_dfs).sort("index").get_column("split")
        return df.sort("stem").with_columns(split_series)
    
    def random_split(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.split(df, random=True)


class NiaDataPathProvider:
    def __init__(
            self,
            reader: NiaDataPathExtractor,
            splitter: DataFrameSplitter,
            channels: list[str],
        ) -> None:
        self.reader = reader
        self.splitter = splitter
        self.channels = channels
    
    def get_split_data_list(self, split: str) -> list[str]:
        df = self.reader.filter_channels(self.channels)
        df = self.splitter.random_split(df)

        df = df.filter(pl.col("split") == split)

        return df.select(
            "collection_path",
            "annotation_path",
        ).rows()
