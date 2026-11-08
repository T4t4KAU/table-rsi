import json

import pytest

from tablersi.partition import partition, write_json


@pytest.fixture
def source_data(tmp_path):
    source = tmp_path / "source"
    tasks = []
    for i, split in enumerate(("proposal", "dev", "test", "test", "test")):
        group = str(2 if i == 3 else i)
        tasks.append(
            {
                "id": str(i),
                "split": split,
                "group": group,
                "cluster": group,
                "questions": ["Question"],
                "supported": i != 4,
            }
        )
        if i != 4:
            write_json(
                source / "stores" / group / "catalog.json",
                [{"content_sha256": group}],
            )
    write_json(source / "public.json", tasks)
    write_json(
        source / "oracle.json",
        {str(i): [{"score_points": [f"REFERENCE_{i}"]}] for i in range(5)},
    )
    write_json(source / "manifest.json", {})
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {"partitions": {"test": {"ids": ["2"]}}})
    return source, protocol


def test_partition_keeps_all_tasks_and_separates_references(source_data, tmp_path):
    source, protocol = source_data
    out = tmp_path / "partition"
    manifest = partition(source, protocol, out)
    assert manifest["partitions"]["postdeploy_test"]["dialogs"] == 3
    assert manifest["partitions"]["postdeploy_test"]["unsupported_dialogs"] == 1
    test = json.loads((out / "postdeploy_test/public.json").read_text())
    assert [t["previous_terminal_group"] for t in test] == [True, True, False]
    assert [t["previous_terminal_dialog"] for t in test] == [True, False, False]
    assert not (out / "postdeploy_test/oracle.json").exists()
    for path in (out / "postdeploy_test").rglob("*.json"):
        assert "REFERENCE_" not in path.read_text()
    oracle = json.loads((out / "scorer_only/oracle.json").read_text())
    assert set(oracle) == {"2", "3", "4"}
    train = json.loads((out / "predeploy_train/oracle.json").read_text())
    assert set(train) == {"0"}
    with pytest.raises(FileExistsError):
        partition(source, protocol, out)


def test_partition_rejects_content_leakage(source_data, tmp_path):
    source, protocol = source_data
    write_json(source / "stores/0/catalog.json", [{"content_sha256": "2"}])
    with pytest.raises(ValueError, match="Identical sheet"):
        partition(source, protocol, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_partition_rejects_old_test_in_training(source_data, tmp_path):
    source, protocol = source_data
    tasks = json.loads((source / "public.json").read_text())
    tasks[2]["split"] = "proposal"
    write_json(source / "public.json", tasks)
    with pytest.raises(ValueError, match="previously tested"):
        partition(source, protocol, tmp_path / "out")
