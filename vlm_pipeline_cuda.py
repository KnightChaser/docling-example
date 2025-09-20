# vlm_pipeline_cuda.py
# NOTE:
# The functionality is the same as vlm_pipeline.py, but this script
# will use CUDA if a GPU is available.
import os
import warnings
import torch
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions,
    InferenceFramework,
    TransformersModelType,
    ResponseFormat,
)
from docling_core.types.doc.base import ImageRefMode
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

cpu_count = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(cpu_count))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_count))
os.environ.setdefault("MKL_NUM_THREADS", str(cpu_count))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_count))

# NOTE: Silence the pin_memory warning from torch DataLoader (CPU-only runs)
warnings.filterwarnings(
    "ignore",
    message=".*'pin_memory' argument is set as true.*",
    category=UserWarning,
    module="torch.utils.data.dataloader",
)

SRC = Path("./test_pdf/cwst2025.pdf")
OUT = Path("output_vlm")
OUT.mkdir(parents=True, exist_ok=True)

vlm_inline = InlineVlmOptions(
    repo_id="ibm-granite/granite-docling-258M",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS,
    transformers_model_type=TransformersModelType.AUTOMODEL_VISION2SEQ,
    supported_devices=[AcceleratorDevice.CUDA, AcceleratorDevice.CPU], # NOTE: Support both CUDA and CPU
    scale=1.5,
    temperature=0.0,
)

# NOTE: Use CUDA if the GPU is available
accel = AcceleratorOptions(
    num_threads=cpu_count,
    device=AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU,
)

vlm_opts = VlmPipelineOptions(
    vlm_options=vlm_inline,
    accelerator_options=accel,
    generate_page_images=True
)

conv = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=vlm_opts,
        )
    }
)

res = conv.convert(SRC)

pages = sorted(res.document.pages.keys())
for page in pages:
    img_dir = OUT / "images" / f"page_{page:03d}"
    img_dir.mkdir(parents=True, exist_ok=True)
    md_path = OUT / f"page_{page:03d}.md"
    res.document.save_as_markdown(
        md_path,
        image_mode=ImageRefMode.REFERENCED,  # Use link instead of base64
        artifacts_dir=img_dir,
        page_no=page,  # per-page export
    )

print("Done. Markdown: output/output_#.md  |  Images under output/images/p###/")

