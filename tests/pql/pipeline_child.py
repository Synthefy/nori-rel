"""Load a saved pipeline in a fresh process and predict with it.

Usage: pipeline_child.py PIPELINE CHECKPOINT SCHEMA ANCHOR_TIME OUTPUT
"""

import json
import sys
from datetime import datetime

import pipeline_fakes

from nori_rel import EnvironmentCredential, NoriRelClient, NoriRelPipeline


def main(pipeline_path, checkpoint, schema, anchor_time, output) -> None:
    pipeline_fakes.install(checkpoint)
    pipeline = NoriRelPipeline.load(pipeline_path)
    cutoff = datetime.fromisoformat(anchor_time)
    with NoriRelClient(device="cpu") as client:
        source = client.connect(
            engine="postgresql",
            credential=EnvironmentCredential(variable="NORI_REL_TEST_DATABASE_URL"),
            schema_name=schema,
            tables=[table.name for table in pipeline.manifest.source.tables],
            time_columns={
                table.name: table.time_column
                for table in pipeline.manifest.source.tables
                if table.time_column is not None
            },
        )
        details = client.predict(
            source=source,
            pipeline=pipeline,
            as_of=cutoff,
            observed_through=cutoff,
            return_details=True,
        )
    frame = details["predictions"]
    result = {
        "entity_id": [int(value) for value in frame["entity_id"]],
        "prediction": frame["prediction"].tolist(),
        "metadata": details["metadata"],
        "inputs": pipeline_fakes.last_inputs_hash(),
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, default=str)


if __name__ == "__main__":
    main(*sys.argv[1:])
