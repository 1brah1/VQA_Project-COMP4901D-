## Eval set (navigation)

Put a small set of images under:
- `data/eval/images/crosswalk/`
- `data/eval/images/stairs/`
- `data/eval/images/obstacles/`

Then describe the expected label(s) in `data/eval/labels.json`.

### `labels.json` schema

```json
{
  "version": 1,
  "items": [
    {
      "id": "crosswalk_001",
      "path": "data/eval/images/crosswalk/crosswalk_001.jpg",
      "task": "crosswalk_signal",
      "labels": { "walk_signal": "red" }
    },
    {
      "id": "stairs_001",
      "path": "data/eval/images/stairs/stairs_001.jpg",
      "task": "stairs",
      "labels": { "stairs_present": "yes" }
    },
    {
      "id": "obstacle_001",
      "path": "data/eval/images/obstacles/obstacle_001.jpg",
      "task": "obstacles",
      "labels": { "obstacle_present": "yes" }
    }
  ]
}
```

