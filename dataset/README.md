# Cattle Video Evaluation Dataset

Expected layout:
```
dataset/
├── <cow_id_1>/
│   ├── video_1.mp4
│   └── video_2.mp4
└── <cow_id_2>/
    ├── video_1.mp4
    └── video_2.mp4
```

To run evaluation on real videos or generate synthetic validation data:
```bash
python3 tools/eval_matching.py --synthetic-demo
```
