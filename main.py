import os
import warnings
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.base import ImageRefMode

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
OUT = Path("output")
OUT.mkdir(parents=True, exist_ok=True)

# NOTE: keep/generate images so REFERENCED export has real files to link
pipe = PdfPipelineOptions()
pipe.images_scale = 2.0  # nicer resolution
pipe.generate_picture_images = True
pipe.generate_page_images = False  # no full-page images

conv = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe)}
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
