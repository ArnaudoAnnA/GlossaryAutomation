#  Copyright (c) 2026. Politecnico di Torino
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import csv
import json
from pathlib import Path


class InputParser:
    @staticmethod
    def parse_json(path: Path) -> list[str]:
        """
        Parse JSON file and return list of "term: definition" entries.
        One entry is created per definition.

        :param path: Path to JSON file.
        :return: List of formatted entries.
        """
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        parent_terms = data.get("parentTerms", [])
        for item in parent_terms:
            term = item.get("term", "").strip()
            if not term:
                continue

            definitions = item.get("definitions") or []
            if not definitions:
                continue

            for def_obj in definitions:
                def_text = def_obj.get("text", "").strip()
                if def_text:
                    entries.append(f"{term}: {def_text}")
        return entries

    @staticmethod
    def parse_csv(path: Path) -> list[str]:
        """
        Parse CSV file and return list of "term: definition" entries.

        :param path: Path to CSV file.
        :return: List of formatted entries.
        """
        entries = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = (row.get("term") or "").strip()
                definition = (row.get("definition") or "").strip()
                if term and definition:
                    entries.append(f"{term}: {definition}")
        return entries
