# Server API

mlx-vc provides a FastAPI server for voice conversion over HTTP.

## Start Server

```bash
python -m mlx_vc.server
# or with options
python -m mlx_vc.server --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /v1/audio/convert

Convert audio using any supported model.

**Request** (multipart/form-data):

| Field | Type | Description |
|-------|------|-------------|
| `source` | file | Source audio file |
| `reference` | file | Reference speaker audio |
| `model` | string | Model name (default: "openvoice") |

**Response**: WAV audio file

**Example**:

```bash
curl -X POST http://localhost:8000/v1/audio/convert \
  -F "source=@my_voice.wav" \
  -F "reference=@target_speaker.wav" \
  -F "model=openvoice" \
  --output converted.wav
```

### GET /v1/models

List available models.

### GET /health

Health check.
