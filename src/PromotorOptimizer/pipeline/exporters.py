# pipeline/exporters.py

import json
from dataclasses import asdict


class ExperimentExporter:

    @staticmethod
    def export_json(experiment, output_path: str):

        with open(output_path, "w") as f:
            json.dump(asdict(experiment), f, indent=4)