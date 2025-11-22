# Pipe resonance estimation from videos

## Usage

Manage dependencies:

```bash
make create-env # create conda env called pipe_resonance
make update-env # update conda env called pipe_resonance if already exists
conda activate pipe_resonance # activate environment
```

Usage

```bash
python main.py "data/WhatsApp Video 2025-04-19 at 20.04.04.mp4"
python main.py "data/WhatsApp Video 2025-04-19 at 20.06.10.mp4"
python main.py "data/IMG_3128.MOV"
python main.py "data/IMG_3129.MOV"
```

Convert output video to whatsapp compatible

```bash
ffmpeg -i <in_video>.mp4 -vcodec libx264 -pix_fmt yuv420p -preset veryfast -crf 23 -movflags +faststart <out_video>.mp4
```

## Laundry list
- [ ] How to run it in windows
- [ ] make `video` arg optional, and if `None` add a "open file" dialog
- [ ] overlay the vertical gate
