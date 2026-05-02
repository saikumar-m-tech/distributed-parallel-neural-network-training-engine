from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
import os

W, H = letter

# ── palette ──────────────────────────────────────────────────────────────────
BLUE        = colors.HexColor("#1A56DB")
BLUE_LIGHT  = colors.HexColor("#E3F0FF")
GREEN       = colors.HexColor("#0E6245")
GREEN_LIGHT = colors.HexColor("#E6F9F1")
AMBER       = colors.HexColor("#92400E")
AMBER_LIGHT = colors.HexColor("#FFFBEB")
RED         = colors.HexColor("#7F1D1D")
RED_LIGHT   = colors.HexColor("#FEF2F2")
PURPLE      = colors.HexColor("#4C1D95")
PURPLE_LIGHT= colors.HexColor("#EDE9FE")
ORANGE      = colors.HexColor("#9A3412")
ORANGE_LIGHT= colors.HexColor("#FFF7ED")
GRAY        = colors.HexColor("#F3F4F6")
GRAY_MID    = colors.HexColor("#6B7280")
GRAY_DARK   = colors.HexColor("#374151")
GRAY_BORDER = colors.HexColor("#D1D5DB")
WHITE       = colors.white
BLACK       = colors.HexColor("#111827")
CODE_BG     = colors.HexColor("#1E1E1E")
CODE_GREEN  = colors.HexColor("#86EFAC")
CODE_GRAY   = colors.HexColor("#D4D4D4")
CODE_COMMENT= colors.HexColor("#6A9955")
CODE_YELLOW = colors.HexColor("#DCDCAA")

BD  = {"style": "SINGLE", "width": 0.5, "color": GRAY_BORDER}

# ── styles ────────────────────────────────────────────────────────────────────
def S(name, **kw):
    base = dict(fontName='Helvetica', fontSize=9.5, textColor=BLACK,
                leading=14, spaceBefore=4, spaceAfter=4)
    base.update(kw)
    return ParagraphStyle(name, **base)

H1S = S('h1', fontName='Helvetica-Bold', fontSize=20, textColor=BLUE,
         spaceBefore=20, spaceAfter=8, leading=24)
H2S = S('h2', fontName='Helvetica-Bold', fontSize=13, textColor=BLUE,
         spaceBefore=14, spaceAfter=5, leading=17)
H3S = S('h3', fontName='Helvetica-Bold', fontSize=10.5, textColor=GRAY_DARK,
         spaceBefore=10, spaceAfter=4, leading=14)
BODY = S('body', alignment=TA_JUSTIFY)
BODYL= S('bodyl')
BLT  = S('blt', leftIndent=14)
CODESML = S('codesml', fontName='Courier', fontSize=8, textColor=CODE_GRAY,
             backColor=CODE_BG, leading=11, leftIndent=8)
PROMPTSML = S('promptsml', fontName='Courier', fontSize=8, textColor=CODE_GREEN,
               backColor=CODE_BG, leading=11, leftIndent=8)
CODELABEL = S('codelabel', fontName='Helvetica-Bold', fontSize=7.5,
               textColor=GRAY_MID, backColor=CODE_BG, leading=10, leftIndent=8)
THDR = S('thdr', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE, leading=12)
TCELL= S('tcell', fontSize=8.5, leading=12, spaceBefore=1, spaceAfter=1)
TCELB= S('tcelb', fontName='Helvetica-Bold', fontSize=8.5, textColor=BLUE, leading=12)
COVER_TITLE = S('ct', fontName='Helvetica-Bold', fontSize=38, textColor=BLUE, leading=44)
COVER_SUB   = S('cs', fontName='Helvetica', fontSize=15, textColor=GRAY_DARK, leading=19)
COVER_META  = S('cm', fontName='Helvetica', fontSize=10, textColor=GRAY_DARK, leading=16)

def sp(n=8):  return Spacer(1, n)
def hr(c=GRAY_BORDER, t=0.5): return HRFlowable(width="100%", thickness=t, color=c, spaceAfter=6, spaceBefore=6)
def hr_blue(): return HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=8, spaceBefore=2)

def h1(t): return [Paragraph(t, H1S), hr_blue()]
def h2(t): return Paragraph(t, H2S)
def h3(t): return Paragraph(t, H3S)
def body(t): return Paragraph(t, BODY)
def bodyl(t): return Paragraph(t, BODYL)
def blt(t): return Paragraph(f"• {t}", BLT)

def code_block(lines, label=None):
    blocks = []
    if label:
        blocks.append(Paragraph(f"  {label}", CODELABEL))
    for line in lines:
        blocks.append(Paragraph(f"  {line}" if line else " ", CODESML))
    blocks.append(sp(6))
    return blocks

def prompt_block(lines, label="Copilot prompt — paste this exactly:"):
    blocks = [Paragraph(f"  {label}", CODELABEL)]
    for line in lines:
        blocks.append(Paragraph(f"  {line}" if line else " ", PROMPTSML))
    blocks.append(sp(6))
    return blocks

def callout(label, text, color, bg, bold_label=True):
    lbl = f"<b>{label}</b>" if bold_label else label
    return [
        Paragraph(lbl, ParagraphStyle('cl', fontName='Helvetica-Bold',
            fontSize=9, textColor=color, leading=13, spaceAfter=3, backColor=bg,
            leftIndent=6, rightIndent=6)),
        Paragraph(text, ParagraphStyle('cb', fontName='Helvetica', fontSize=9,
            textColor=BLACK, leading=13, backColor=bg, leftIndent=6, rightIndent=6)),
        sp(6),
    ]

def severity_badge(sev):
    colors_map = {
        'CRITICAL': (RED, RED_LIGHT, "🔴  CRITICAL"),
        'HIGH':     (ORANGE, ORANGE_LIGHT, "🟠  HIGH"),
        'MEDIUM':   (AMBER, AMBER_LIGHT, "🟡  MEDIUM"),
        'LOW':      (GREEN, GREEN_LIGHT, "🟢  LOW"),
    }
    c, bg, txt = colors_map.get(sev, (GRAY_DARK, GRAY, sev))
    p = Paragraph(txt, ParagraphStyle('sb', fontName='Helvetica-Bold',
        fontSize=9, textColor=c, leading=12))
    t = Table([[p]], colWidths=[1.1*inch], repeatRows=1, splitByRow=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),bg),
        ('BOX',(0,0),(-1,-1),0.5,c),
        ('LEFTPADDING',(0,0),(-1,-1),6), ('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),3), ('BOTTOMPADDING',(0,0),(-1,-1),3),
    ]))
    return t

def issue_card(num, severity, title, files, problem, what_happens, fix_description, prompt_lines, verify_items):
    sev_colors = {
        'CRITICAL': (RED, RED_LIGHT),
        'HIGH':     (ORANGE, ORANGE_LIGHT),
        'MEDIUM':   (AMBER, AMBER_LIGHT),
        'LOW':      (GREEN, GREEN_LIGHT),
    }
    sc, sbg = sev_colors.get(severity, (GRAY_DARK, GRAY))

    content = []

    # header
    hdr_data = [[
        Paragraph(f"<b>Issue {num}</b>", ParagraphStyle('ih', fontName='Helvetica-Bold',
            fontSize=10, textColor=WHITE, leading=13)),
        Paragraph(f"<b>{title}</b>", ParagraphStyle('it', fontName='Helvetica-Bold',
            fontSize=11, textColor=WHITE, leading=14)),
        Paragraph(severity, ParagraphStyle('is', fontName='Helvetica-Bold',
            fontSize=9, textColor=sbg, leading=12)),
    ]]
    hdr = Table(hdr_data, colWidths=[0.65*inch, 4.5*inch, 1.35*inch], repeatRows=1, splitByRow=1)
    hdr.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),sc),
        ('LEFTPADDING',(0,0),(-1,-1),10), ('RIGHTPADDING',(0,0),(-1,-1),10),
        ('TOPPADDING',(0,0),(-1,-1),9), ('BOTTOMPADDING',(0,0),(-1,-1),9),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('BACKGROUND',(2,0),(2,0),colors.HexColor("#FFFFFF33")),
    ]))
    content.append(hdr)

    # files affected
    file_str = "  ".join([f"<font name='Courier' size='8'>{f}</font>" for f in files])
    content.append(Paragraph(f"<b>Files:</b>  {file_str}",
        ParagraphStyle('ff', fontName='Helvetica', fontSize=8.5,
        textColor=GRAY_DARK, leading=13, backColor=GRAY,
        leftIndent=0, spaceAfter=0, spaceBefore=0)))

    # problem
    content.append(Paragraph("<b>What is wrong:</b>", ParagraphStyle('pw',
        fontName='Helvetica-Bold', fontSize=9, textColor=BLACK, leading=12,
        spaceBefore=8, spaceAfter=3)))
    content.append(Paragraph(problem, ParagraphStyle('pb', fontName='Helvetica',
        fontSize=9, textColor=BLACK, leading=13, spaceAfter=4)))

    # what happens
    content.append(Paragraph("<b>Impact if left unfixed:</b>", ParagraphStyle('pw2',
        fontName='Helvetica-Bold', fontSize=9, textColor=sc, leading=12,
        spaceAfter=3)))
    content.append(Paragraph(what_happens, ParagraphStyle('wi', fontName='Helvetica',
        fontSize=9, textColor=BLACK, leading=13, spaceAfter=4)))

    # fix description
    content.append(Paragraph("<b>Fix:</b>", ParagraphStyle('fd',
        fontName='Helvetica-Bold', fontSize=9, textColor=GREEN, leading=12,
        spaceAfter=3)))
    content.append(Paragraph(fix_description, ParagraphStyle('fi', fontName='Helvetica',
        fontSize=9, textColor=BLACK, leading=13, spaceAfter=6)))

    # copilot prompt
    if prompt_lines:
        content.extend(prompt_block(prompt_lines))

    # verification
    if verify_items:
        content.append(Paragraph("<b>Verify before moving on:</b>",
            ParagraphStyle('vl', fontName='Helvetica-Bold', fontSize=9,
            textColor=GREEN, leading=12, spaceBefore=6, spaceAfter=3)))
        for item in verify_items:
            content.append(Paragraph(f"☐ {item}", ParagraphStyle('vi',
                fontName='Helvetica', fontSize=8.5, textColor=BLACK, leading=12,
                leftIndent=12, spaceAfter=2)))

    return content + [sp(16)]

def tH(cols, ws):
    return Table([[Paragraph(c, THDR) for c in cols]], colWidths=ws, repeatRows=1, splitByRow=1,
        style=TableStyle([
            ('BACKGROUND',(0,0),(-1,-1),BLUE),
            ('LEFTPADDING',(0,0),(-1,-1),8), ('RIGHTPADDING',(0,0),(-1,-1),8),
            ('TOPPADDING',(0,0),(-1,-1),6), ('BOTTOMPADDING',(0,0),(-1,-1),6),
        ]))

def data_table(headers, rows, col_widths):
    data = [[Paragraph(h, THDR) for h in headers]]
    for i, row in enumerate(rows):
        data.append([Paragraph(str(c), TCELL) if isinstance(c,str) else c for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1, splitByRow=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),BLUE),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE, GRAY]),
        ('GRID',(0,0),(-1,-1),0.3,GRAY_BORDER),
        ('LEFTPADDING',(0,0),(-1,-1),8), ('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
    ]))
    return t

def phase_banner(txt, bg):
    t = Table([[Paragraph(txt, ParagraphStyle('pb', fontName='Helvetica-Bold',
        fontSize=14, textColor=WHITE, leading=18))]], colWidths=[6.5*inch], repeatRows=1, splitByRow=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),bg),
        ('LEFTPADDING',(0,0),(-1,-1),16), ('TOPPADDING',(0,0),(-1,-1),10),
        ('BOTTOMPADDING',(0,0),(-1,-1),10),
    ]))
    return t

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BLUE)
    canvas.setLineWidth(1.5)
    canvas.line(0.75*inch, H-0.55*inch, W-0.75*inch, H-0.55*inch)
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(GRAY_MID)
    canvas.drawString(0.75*inch, H-0.45*inch, "ParallelNet — Code Review & Fix Guide")
    canvas.drawRightString(W-0.75*inch, H-0.45*inch, "Leo | UL MSc SE 2025/26")
    canvas.setStrokeColor(GRAY_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(0.75*inch, 0.55*inch, W-0.75*inch, 0.55*inch)
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(GRAY_MID)
    canvas.drawCentredString(W/2, 0.38*inch, f"Page {doc.page}")
    canvas.restoreState()

# ═══════════════════════════════════════════════════════════════════════════════
def build():
    path = "A:/distributed-parallel-neural-network-training-engine/ParallelNet_Code_Review.pdf"
    doc = SimpleDocTemplate(path, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.8*inch, bottomMargin=0.7*inch)
    story = []

    # ── COVER ──────────────────────────────────────────────────────────────────
    story += [
        sp(30),
        Paragraph("ParallelNet", COVER_TITLE),
        Paragraph("Code Review — Bugs, Vulnerabilities & Fix Guide", COVER_SUB),
        Paragraph("With exact Copilot prompts for every fix", ParagraphStyle(
            'ci', fontName='Helvetica-Oblique', fontSize=11, textColor=GRAY_MID, leading=15)),
        hr_blue(),
        sp(8),
        Paragraph("<b>Reviewed by:</b> Claude Sonnet 4.5", COVER_META),
        Paragraph("<b>Codebase:</b> ParallelNet — distributed neural network training engine", COVER_META),
        Paragraph("<b>Files reviewed:</b> 20 source files across kernels/, engine/, mpi/, bridge/, python/, tests/, benchmarks/", COVER_META),
        sp(16),
        *callout("Headline finding — read this first:",
            "The CUDA kernels in matmul.cu, activations.cu, and sgd_update.cu are completely disconnected from the training engine. dense_layer.cpp and network.cpp copy all data to the CPU, compute everything in C++ for-loops, and copy back. Your GPU is used only as a memory store — not for any arithmetic. The benchmark in kernel_bench.py tests its own inline kernel code, not your project's kernels. Issues 1 and 2 in this document fix both of these. Everything else in the codebase is solid — the architecture is correct, the build system works, and the test infrastructure is good. This is a wiring problem, not a design problem.",
            RED, RED_LIGHT),
        sp(16),
        Paragraph("<b>Issue summary:</b>", COVER_META),
        sp(8),
        data_table(
            ["#", "Severity", "Title", "Files"],
            [
                ["1", Paragraph("🔴 CRITICAL", ParagraphStyle('r', fontName='Helvetica-Bold', fontSize=8.5, textColor=RED, leading=12)), "Dense layer forward/backward is 100% CPU", "engine/dense_layer.cpp"],
                ["2", Paragraph("🔴 CRITICAL", ParagraphStyle('r', fontName='Helvetica-Bold', fontSize=8.5, textColor=RED, leading=12)), "Network forward/backward is 100% CPU", "engine/network.cpp"],
                ["3", Paragraph("🔴 CRITICAL", ParagraphStyle('r', fontName='Helvetica-Bold', fontSize=8.5, textColor=RED, leading=12)), "MPI disabled — Trainer never syncs gradients", "setup.py, bridge/bindings.cpp"],
                ["4", Paragraph("🟠 HIGH", ParagraphStyle('o', fontName='Helvetica-Bold', fontSize=8.5, textColor=ORANGE, leading=12)), "Benchmark tests its own kernels, not yours", "benchmarks/kernel_bench.py"],
                ["5", Paragraph("🟠 HIGH", ParagraphStyle('o', fontName='Helvetica-Bold', fontSize=8.5, textColor=ORANGE, leading=12)), "CUDA_CHECK in GpuBuffer destructor", "kernels/kernel_utils.cuh"],
                ["6", Paragraph("🟡 MEDIUM", ParagraphStyle('a', fontName='Helvetica-Bold', fontSize=8.5, textColor=AMBER, leading=12)), "SGD test does not chain updates", "tests/test_kernels.py"],
                ["7", Paragraph("🟡 MEDIUM", ParagraphStyle('a', fontName='Helvetica-Bold', fontSize=8.5, textColor=AMBER, leading=12)), "Root CMakeLists.txt is empty", "CMakeLists.txt"],
                ["8", Paragraph("🟡 MEDIUM", ParagraphStyle('a', fontName='Helvetica-Bold', fontSize=8.5, textColor=AMBER, leading=12)), "Three stub files never implemented", "ring_allreduce.cpp, conv_layer.cpp, plot_results.py"],
                ["9", Paragraph("🟢 LOW", ParagraphStyle('g', fontName='Helvetica-Bold', fontSize=8.5, textColor=GREEN, leading=12)), "GpuBuffer destructor prints pollute test output", "kernels/kernel_utils.cuh"],
                ["10", Paragraph("🟢 LOW", ParagraphStyle('g', fontName='Helvetica-Bold', fontSize=8.5, textColor=GREEN, leading=12)), "cupy DLL setup runs after cupy is imported", "benchmarks/kernel_bench.py"],
            ],
            [0.3*inch, 1.0*inch, 2.8*inch, 2.4*inch]
        ),
        PageBreak(),
    ]

    # ── SECTION 1: CRITICAL ISSUES ─────────────────────────────────────────────
    story += h1("Critical Issues — Fix These First")

    # ISSUE 1
    story.extend(issue_card(
        1, "CRITICAL",
        "Dense layer forward and backward are 100% CPU",
        ["engine/dense_layer.cpp", "kernels/matmul.cu", "kernels/activations.cu"],
        "dense_layer.cpp::forward() copies input weights and bias to std::vector on the CPU, runs a triple nested for-loop for the matrix multiply and ReLU, then copies the result back to GPU memory. dense_layer.cpp::backward() does the same for dW, db, and grad_in. The kernels in matmul.cu (naive_matmul_gpu, tiled_matmul_gpu) and activations.cu (relu_forward, relu_backward) are never called anywhere in the engine.",
        "Your GPU does zero arithmetic during training. All floating point work runs on 6 CPU cores instead of 1024 CUDA cores. A Dense(3072, 512) forward pass does 3072 * 512 = 1.57M multiplications per sample — on CPU this is orders of magnitude slower than using your CUDA kernels. The benchmark numbers in gtx1650_results.json are meaningless for the actual training speed.",
        "Rewrite forward() to call tiled_matmul_gpu() followed by relu_forward kernel. Rewrite backward() to call matmul for dW and grad_in, then relu_backward kernel. The GpuBuffer data pointers are already on the device — you just need to pass them to the kernel launches.",
        [
            "Rewrite engine/dense_layer.cpp to use GPU kernels throughout.",
            "Do NOT change the function signatures or the .hpp file.",
            "",
            "forward(const FloatBuffer& in, FloatBuffer& out):",
            "  // Step 1: output_pre_relu = in @ weights.T + bias",
            "  //   Use tiled_matmul_gpu(in.data(), weights_.data(), pre_relu.data(),",
            "  //                        batch, out_features_, in_features_)",
            "  //   then add bias: launch a simple broadcast-add kernel",
            "  // Step 2: out = relu(output_pre_relu)",
            "  //   Use relu_forward kernel from activations.cu",
            "  // Save: last_input_ = copy of in.data() (needed for backward)",
            "  //        last_pre_relu_ = copy of pre_relu (needed for relu_backward)",
            "",
            "backward(const FloatBuffer& grad_out, FloatBuffer& grad_in):",
            "  // Step 1: relu_backward(last_pre_relu_.data(), grad_out.data(), n)",
            "  //   This zeros out grad_out where pre_relu <= 0 (in-place on a copy)",
            "  // Step 2: dW = grad_after_relu.T @ last_input_",
            "  //   Use tiled_matmul_gpu",
            "  // Step 3: db = sum grad_after_relu over batch (use a reduction kernel)",
            "  // Step 4: grad_in = grad_after_relu @ weights_",
            "  //   Use tiled_matmul_gpu",
            "",
            "Add declarations for tiled_matmul_gpu and relu kernels in matmul.cu and",
            "activations.cu. Also add a bias_add kernel to activations.cu:",
            "  __global__ void bias_add(float* out, const float* bias,",
            "                           int batch, int features)",
            "",
            "Add a db_reduce kernel to activations.cu:",
            "  __global__ void sum_over_batch(const float* grad, float* db,",
            "                                  int batch, int features)",
            "  Each thread handles one feature, sums over all batch items.",
        ],
        [
            "After the fix: add a print in forward() that logs 'running GPU forward'. Verify it prints during training — not the old CPU log.",
            "Run test_bridge.py — loss must still decrease (proves backward pass is correct through the new kernel path)",
            "Run nvidia-smi dmon -s u during training — GPU utilisation should now show >0% (was 0% before this fix)",
            "Time one epoch before and after — should be faster for batch_size >= 64",
        ]
    ))

    # ISSUE 2
    story.extend(issue_card(
        2, "CRITICAL",
        "Network forward and backward are 100% CPU",
        ["engine/network.cpp"],
        "network.cpp::forward() computes the output layer logits using a triple nested CPU for-loop, then runs softmax and cross-entropy loss entirely in CPU std::vector code. network.cpp::backward() computes dout_weights_, dout_bias_, and grad_hidden_ using CPU loops. The CUDA softmax_forward and cross_entropy_loss kernels in activations.cu are never called.",
        "Same as Issue 1 — the output layer (hidden→logits) runs on CPU. Combined with Issue 1, the entire forward and backward pass is CPU. The only GPU operations currently are cudaMalloc and cudaMemcpy — you are paying the PCIe transfer cost with none of the compute benefit.",
        "Rewrite the output layer forward pass in network.cpp to use tiled_matmul_gpu for the logits computation and softmax_forward for the probabilities. Rewrite the backward pass to use CUDA kernels for the gradient computation. The hidden layer outputs are already in a FloatBuffer — pass hidden_buf.data() directly to the kernel.",
        [
            "Rewrite network.cpp forward pass output layer to use GPU kernels.",
            "Do NOT change the public interface of Network.",
            "",
            "In Network::forward(), after calling layers_[0].forward(input_buf, hidden_buf):",
            "  // Output layer logits: logits = hidden @ out_weights_.T + out_bias_",
            "  // Store out_weights_ and out_bias_ as FloatBuffers, not std::vector",
            "  // Use tiled_matmul_gpu(hidden_buf.data(), out_weights_gpu_.data(),",
            "  //                      logits_gpu_.data(), batch, out_features_, hidden_features_)",
            "  // Then: softmax_forward<<<batch, 256, 256*4>>>(logits_gpu_.data(),",
            "  //                         probs_gpu_.data(), batch, out_features_)",
            "",
            "In Network::backward():",
            "  // grad_logits[b][o] = probs[b][o] - (o == label[b] ? 1 : 0) / batch",
            "  // Launch a kernel for this — one thread per (b, o) pair",
            "  // dout_weights_ = grad_logits.T @ hidden (tiled_matmul_gpu)",
            "  // dout_bias_ = sum(grad_logits, axis=batch) (sum_over_batch kernel)",
            "  // grad_hidden = grad_logits @ out_weights_ (tiled_matmul_gpu)",
            "  // Pass grad_hidden to layers_[0].backward()",
            "",
            "Change out_weights_ and out_bias_ from std::vector<float> to FloatBuffer.",
            "Update save_weights() and load_weights() to use copy_to_host/copy_from_host.",
        ],
        [
            "Run test_network.cpp — initial loss must still be between 2.1 and 2.5",
            "Loss must reach below 1.0 in 50 steps (proves GPU backward pass is correct)",
            "run nvidia-smi dmon -s u — GPU utilisation visibly non-zero during training",
            "compare accuracy after both fixes against the PyTorch baseline in python/ — should be within 5%",
        ]
    ))

    # ISSUE 3
    story.extend(issue_card(
        3, "CRITICAL",
        "MPI is fully disabled — Trainer never syncs gradients",
        ["setup.py", "bridge/bindings.cpp"],
        "setup.py defines PARALLELNET_NO_MPI=1 as a preprocessor macro. This activates the stub GradientSync in gradient_sync.hpp where allreduce_mean() is a no-op (empty function body). Additionally, bridge/bindings.cpp::Trainer::train_step() calls net_.sgd_step(learning_rate_, nullptr) — always passing nullptr for the sync pointer, meaning even if MPI were enabled, gradients would never be averaged. Running mpirun -n 2 python train.py produces two workers that each train independently on their own shard without ever synchronising — the weights diverge, not converge.",
        "Distributed training silently does nothing. You can run mpirun -n 2 and observe two workers producing completely different loss values by epoch 2 — proof that gradients are not being averaged. Every benchmark result from multi-worker runs in your dissertation is invalid until this is fixed.",
        "Remove PARALLELNET_NO_MPI from setup.py. Add MPI libraries to the pybind11 extension link libraries. In bindings.cpp, construct a GradientSync member in Trainer and pass it to sgd_step(). The GradientSync constructor calls MPI_Init — ensure this happens exactly once by checking MPI_Initialized first (which gradient_sync.hpp already does correctly).",
        [
            "Fix setup.py to enable MPI in the Python extension build.",
            "Fix bindings.cpp Trainer to construct and use GradientSync.",
            "",
            "In setup.py:",
            "  1. Remove define_macros=[('PARALLELNET_NO_MPI', '1')]",
            "  2. Add MPI include and library dirs:",
            "     import subprocess",
            "     mpi_compile = subprocess.check_output(['mpicc','--showme:compile']).decode().split()",
            "     mpi_link    = subprocess.check_output(['mpicc','--showme:link']).decode().split()",
            "     # Parse -I, -L, -l flags and add to include_dirs, library_dirs, libraries",
            "  3. Add 'mpi/gradient_sync.cpp' to sources list",
            "",
            "In bridge/bindings.cpp:",
            "  1. Add #include '../mpi/gradient_sync.hpp'",
            "  2. Add GradientSync sync_; as a private member of Trainer",
            "  3. Change train_step to call: net_.sgd_step(learning_rate_, &sync_);",
            "  4. In Trainer constructor, initialise GradientSync before Network",
            "     (MPI_Init must run before any MPI call)",
            "",
            "On Windows with MS-MPI, library_dirs should include:",
            "  os.environ.get('MSMPI_LIB64', 'C:/Program Files/Microsoft MPI/Lib/x64')",
            "and libraries=['msmpi']",
        ],
        [
            "mpirun -n 2 python python/train.py --epochs 5: both workers must print the same loss at each epoch (within 0.01)",
            "mpirun -n 1 python python/train.py: same loss trajectory as before (single-worker unchanged)",
            "Add a print in allreduce_mean showing the gradient buffer checksum before and after — verify it changes on 2-worker runs and is identical on both ranks after the call",
        ]
    ))

    story.append(PageBreak())
    story += h1("High Severity Issues")

    # ISSUE 4
    story.extend(issue_card(
        4, "HIGH",
        "Benchmark tests its own inline kernels, not your project's kernels",
        ["benchmarks/kernel_bench.py"],
        "kernel_bench.py defines KERNEL_CODE as a raw string at the top of the file containing its own separate C++/CUDA implementations of naive_matmul_kernel and tiled_matmul_kernel with extern 'C' qualifiers. It compiles these via cupy.RawModule and benchmarks them. The kernels in your project's matmul.cu are never loaded or called by the benchmark. The results in gtx1650_results.json are for the inline benchmark kernels, not your project's matmul.cu.",
        "Your dissertation Experiment 1 (CUDA kernel vs cuBLAS) is not measuring what you claim. The chart shows performance of benchmark-internal code that is entirely separate from your engine. If you optimise matmul.cu, the benchmark numbers will not change. When an interviewer or dissertation examiner asks you to explain the connection between your benchmark and your engine code, there is no connection.",
        "Replace the inline KERNEL_CODE string with code that loads and compiles your actual matmul.cu file from disk using cupy.RawModule. The extern C qualifiers already present in the benchmark's inline code need to be added to matmul.cu (the project kernels currently lack them). Then the benchmark sources and the engine kernels are identical code.",
        [
            "Rewrite benchmarks/kernel_bench.py to load matmul.cu from disk.",
            "",
            "Replace the KERNEL_CODE string and RawModule construction with:",
            "  from pathlib import Path",
            "  import cupy as cp",
            "",
            "  def load_matmul_module():",
            "      src_path = Path(__file__).resolve().parents[1] / 'kernels' / 'matmul.cu'",
            "      source = src_path.read_text(encoding='utf-8')",
            "      return cp.RawModule(",
            "          code=source,",
            "          options=('--std=c++17',),",
            "          name_expressions=('naive_matmul_kernel', 'tiled_matmul_kernel'),",
            "      )",
            "",
            "Also add extern 'C' to the kernel declarations in kernels/matmul.cu:",
            "  extern 'C' __global__ void naive_matmul_kernel(...)",
            "  extern 'C' __global__ void tiled_matmul_kernel(...)",
            "",
            "Keep KERNEL_CODE removed — do not keep a copy.",
            "The benchmark should now fail to build if matmul.cu has a syntax error,",
            "which is the correct behaviour for a benchmark of your own code.",
        ],
        [
            "Run the benchmark — confirm it loads from disk by checking: add a syntax error to matmul.cu and verify the benchmark raises a compile error",
            "Remove the syntax error. Run benchmark again. Confirm tiled_matmul numbers are within 5% of what gtx1650_results.json shows (the inline and project kernels should produce similar numbers since the algorithm is the same)",
            "The benchmark JSON label should now say your GPU name, not 'current' — update --label default to read from nvidia-smi",
        ]
    ))

    # ISSUE 5
    story.extend(issue_card(
        5, "HIGH",
        "CUDA_CHECK macro used inside GpuBuffer destructor",
        ["kernels/kernel_utils.cuh"],
        "The GpuBuffer destructor calls CUDA_CHECK(cudaFree(data_)) which expands to a macro that calls exit(EXIT_FAILURE) on error. Calling exit() inside a destructor is undefined behaviour in C++ — if the destructor is called during stack unwinding (exception propagation), calling exit() immediately terminates the program without further cleanup, bypassing all other destructors and potentially corrupting state. Additionally, if the CUDA context has already been destroyed (e.g. during program shutdown), cudaFree returns cudaErrorCudartUnloading — CUDA_CHECK then calls exit(), which is the wrong response to a normal shutdown sequence.",
        "Any exception thrown after a GpuBuffer is constructed (e.g. a cudaMemcpy failure in copy_from_host()) causes the destructor to be called during stack unwinding. If cudaFree then fails (unlikely but possible), exit() is called from inside a destructor during stack unwinding — this is undefined behaviour and typically causes process termination with no useful error message. The GpuTimer destructor has the same issue with cudaEventDestroy.",
        "In the destructor, call cudaFree directly without CUDA_CHECK and log any non-success error code to stderr without calling exit(). The destructor should always complete. For GpuTimer, same fix for cudaEventDestroy.",
        [
            "Fix GpuBuffer::~GpuBuffer() and GpuTimer::~GpuTimer() in kernel_utils.cuh.",
            "Replace the destructor bodies with safe versions that do not call exit().",
            "",
            "GpuBuffer destructor — replace with:",
            "  ~GpuBuffer() noexcept {",
            "      if (data_ != nullptr) {",
            "          cudaError_t err = cudaFree(data_);",
            "          if (err != cudaSuccess && err != cudaErrorCudartUnloading) {",
            "              fprintf(stderr, 'GpuBuffer cudaFree failed: %s\\n',",
            "                      cudaGetErrorString(err));",
            "          }",
            "          data_ = nullptr;",
            "      }",
            "  }",
            "",
            "GpuTimer destructor — replace with:",
            "  ~GpuTimer() noexcept {",
            "      cudaEventDestroy(start_event_);",
            "      cudaEventDestroy(stop_event_);",
            "  }",
            "",
            "Also mark GpuBuffer move constructor and move assignment as noexcept.",
            "Do NOT add CUDA_CHECK back to these destructors.",
        ],
        [
            "Write a test that deliberately throws an exception after constructing a GpuBuffer. The program should print the exception message and exit cleanly — NOT terminate with no output.",
            "Run cuda-memcheck on test_network.cpp — no invalid free errors",
            "Verify the fix compiles without warnings with -Wall -Wextra",
        ]
    ))

    story.append(PageBreak())
    story += h1("Medium Severity Issues")

    # ISSUE 6
    story.extend(issue_card(
        6, "MEDIUM",
        "SGD test resets input each iteration — does not chain updates",
        ["tests/test_kernels.py"],
        "test_sgd_update() runs the SGD kernel twice in a loop but reconstructs weights_gpu and gradients_gpu from the original numpy arrays at the start of each iteration. This means both iterations test exactly one step from weights=1.0 and the assertion weights_approx_0.999 passes twice but proves nothing about chaining. A bug where the kernel only updates the first element but not the rest would still pass this test.",
        "The test gives false confidence. If there is an off-by-one bug in the thread indexing (e.g. only updating elements where idx < blockDim.x), the test still passes because both iterations start fresh from 1.0 and check 0.999.",
        "Move the GpuBuffer creation outside the loop and chain two updates: after the first update weights should be 0.999, after the second update starting from 0.999 they should be 0.999 - 0.01*0.1 = 0.998. Test both values. Also add a large-n test (n=100000) to exercise multiple blocks and catch thread indexing bugs.",
        [
            "Fix the test_sgd_update function in tests/test_kernels.py.",
            "",
            "Replace the current test_sgd_update with:",
            "def test_sgd_update(sgd_module):",
            "    n = 2048",
            "    kernel = sgd_module.get_function('sgd_update')",
            "    block = (256, 1, 1)",
            "    grid = (math.ceil(n / block[0]), 1, 1)",
            "    lr = np.float32(0.01)",
            "",
            "    # --- Test 1: single step ---",
            "    w_gpu = cp.full(n, 1.0, dtype=cp.float32)",
            "    g_gpu = cp.full(n, 0.1, dtype=cp.float32)",
            "    kernel(grid, block, (w_gpu, g_gpu, lr, np.int32(n)))",
            "    cp.cuda.runtime.deviceSynchronize()",
            "    assert cp.allclose(w_gpu, cp.full(n, 0.999), rtol=1e-6, atol=1e-6)",
            "",
            "    # --- Test 2: chain two steps from the same buffer ---",
            "    kernel(grid, block, (w_gpu, g_gpu, lr, np.int32(n)))",
            "    cp.cuda.runtime.deviceSynchronize()",
            "    assert cp.allclose(w_gpu, cp.full(n, 0.998), rtol=1e-6, atol=1e-6)",
            "",
            "    # --- Test 3: large n to catch multi-block indexing bugs ---",
            "    n_large = 100_000",
            "    grid_large = (math.ceil(n_large / block[0]), 1, 1)",
            "    w2 = cp.full(n_large, 2.0, dtype=cp.float32)",
            "    g2 = cp.full(n_large, 0.5, dtype=cp.float32)",
            "    kernel(grid_large, block, (w2, g2, lr, np.int32(n_large)))",
            "    cp.cuda.runtime.deviceSynchronize()",
            "    assert cp.allclose(w2, cp.full(n_large, 1.995), rtol=1e-6, atol=1e-6)",
        ],
        [
            "All three test assertions pass",
            "Deliberately break the kernel (change `i += stride` to `i += 1`) and verify test 3 catches it but tests 1 and 2 might not — this confirms test 3 covers multi-block bugs",
        ]
    ))

    # ISSUE 7
    story.extend(issue_card(
        7, "MEDIUM",
        "Root CMakeLists.txt is empty",
        ["CMakeLists.txt"],
        "The root CMakeLists.txt is an empty file. The only working CMakeLists.txt is in kernels/ and only builds test_device.cu. There is no root-level CMake build that compiles the full project (engine, mpi, bridge). The README instructions for building the C++ bridge (pip install -e . using setup.py) work, but cmake at the root does nothing. test_network.cpp and test_mpi.cpp cannot be built with cmake because there is no target for them.",
        "A developer (or dissertation examiner) who follows standard practice and runs cmake .. at the root gets an empty build. The C++ unit tests test_network.cpp and test_mpi.cpp have no way to be compiled. The GpuTimer and matmul kernel benchmarks that require compiled C++ binaries cannot be built.",
        "Write a proper root CMakeLists.txt that builds all targets: the parallelnet_cpp Python extension, test_network, and test_mpi executables. The kernels/ subdirectory CMakeLists.txt can be included via add_subdirectory().",
        [
            "Write a complete root CMakeLists.txt that builds the full project.",
            "",
            "cmake_minimum_required(VERSION 3.24)",
            "project(ParallelNet CUDA CXX)",
            "",
            "set(CMAKE_CXX_STANDARD 17)",
            "set(CMAKE_CXX_STANDARD_REQUIRED ON)",
            "set(CMAKE_CUDA_STANDARD 17)",
            "set(CMAKE_CUDA_ARCHITECTURES 75)  # GTX 1650",
            "",
            "find_package(CUDAToolkit 12.0 REQUIRED)",
            "find_package(MPI REQUIRED)",
            "find_package(Python3 COMPONENTS Interpreter Development REQUIRED)",
            "find_package(pybind11 REQUIRED)",
            "",
            "# CUDA kernel library (shared between engine and tests)",
            "add_library(parallelnet_kernels STATIC",
            "    kernels/matmul.cu",
            "    kernels/activations.cu",
            "    kernels/sgd_update.cu",
            ")",
            "set_target_properties(parallelnet_kernels PROPERTIES",
            "    CUDA_SEPARABLE_COMPILATION ON",
            "    CUDA_ARCHITECTURES 75",
            ")",
            "target_link_libraries(parallelnet_kernels PUBLIC CUDA::cudart)",
            "",
            "# Engine library",
            "add_library(parallelnet_engine STATIC",
            "    engine/dense_layer.cpp",
            "    engine/network.cpp",
            ")",
            "target_link_libraries(parallelnet_engine PUBLIC parallelnet_kernels MPI::MPI_CXX)",
            "target_include_directories(parallelnet_engine PUBLIC kernels/ mpi/)",
            "",
            "# Python extension",
            "pybind11_add_module(parallelnet_cpp bridge/bindings.cpp)",
            "target_link_libraries(parallelnet_cpp PRIVATE parallelnet_engine)",
            "",
            "# C++ test executables",
            "add_executable(test_network tests/test_network.cpp)",
            "target_link_libraries(test_network PRIVATE parallelnet_engine)",
            "",
            "add_executable(test_mpi tests/test_mpi.cpp)",
            "target_link_libraries(test_mpi PRIVATE MPI::MPI_CXX)",
            "target_include_directories(test_mpi PRIVATE mpi/)",
            "",
            "add_subdirectory(kernels)",
        ],
        [
            "cmake -B build . && cmake --build build --parallel 4 completes without errors",
            "./build/test_device prints GTX 1650 device info",
            "./build/test_network runs and prints 'All Dense layer tests passed'",
            "mpirun -n 2 ./build/test_mpi prints [2.0, 3.0, 4.0] on both workers",
        ]
    ))

    # ISSUE 8
    story.extend(issue_card(
        8, "MEDIUM",
        "Three stub files are empty — Ring-AllReduce, Conv layer, Plot results",
        ["mpi/ring_allreduce.cpp", "engine/conv_layer.cpp", "python/plot_results.py"],
        "ring_allreduce.cpp, conv_layer.cpp, and plot_results.py are all empty files. ring_allreduce.cpp is listed as a stretch goal in the spec. conv_layer.cpp is listed in the file structure but never implemented. plot_results.py has no implementation, meaning your dissertation experiment charts have to be generated manually or via the benchmark script — there is no unified plotting pipeline.",
        "Without plot_results.py, reproducing your dissertation charts requires re-running the full training grid manually. Without ring_allreduce.cpp, Experiment stretch goal cannot be run. These gaps are visible in the GitHub repo and will be noticed by anyone who clones it.",
        "Implement plot_results.py first — it is needed for dissertation Experiments 2, 3, and 4. Then implement ring_allreduce.cpp as the stretch goal. conv_layer.cpp can remain a documented future work item if time is limited.",
        [
            "Implement python/plot_results.py with functions for all four dissertation experiments.",
            "",
            "def plot_convergence(csv_path: str, output: str = 'plots/convergence.png'):",
            "    '''Load per-epoch loss and accuracy from CSV, plot convergence curve.'''",
            "",
            "def plot_scaling_efficiency(results: dict, output: str = 'plots/scaling.png'):",
            "    '''",
            "    results = {1: {'time_per_epoch': 30.1, 'final_acc': 0.47},",
            "               2: {'time_per_epoch': 18.3, 'final_acc': 0.46},",
            "               4: {'time_per_epoch': 11.2, 'final_acc': 0.46}}",
            "    Plots: actual speedup bars + ideal linear line + parallel efficiency %.",
            "    '''",
            "",
            "def plot_communication_breakdown(results: dict, output: str):",
            "    '''Stacked bar: compute_time vs sync_time per worker count.'''",
            "",
            "def plot_accuracy_parity(parallelnet_csv: str, pytorch_csv: str, output: str):",
            "    '''Overlay two convergence curves on one chart.'''",
            "",
            "def generate_summary_table(results: dict, output: str):",
            "    '''Write summary_table.csv with columns:",
            "    Engine, Workers, TimePerEpoch, FinalAcc, VsPyTorch.'''",
            "",
            "Use matplotlib and pandas. Include a __main__ block that runs all plots",
            "from a results/ directory.",
        ],
        [
            "python python/plot_results.py --help shows all plot commands",
            "Feed mock CSV data and verify all four chart types save correctly",
            "Charts have axis labels, titles, legends, and tight_layout()",
        ]
    ))

    story.append(PageBreak())
    story += h1("Low Severity Issues")

    # ISSUE 9
    story.extend(issue_card(
        9, "LOW",
        "GpuBuffer destructor prints pollute test output",
        ["kernels/kernel_utils.cuh"],
        "The GpuBuffer destructor contains a print statement: 'GpuBuffer freeing N elements (free #N)'. It is throttled (only prints the first 20 and every 200th after that) but it still produces output during pytest runs, making test output noisy and harder to read. The static counter is also shared across all GpuBuffer instances globally, which is not thread-safe.",
        "Running pytest tests/test_kernels.py produces dozens of 'GpuBuffer freeing...' lines intermixed with test output. If you pipe test results to a log file, these prints make grep for failures harder. The static counter is not atomic — if two GpuBuffers are destroyed concurrently in different threads, the counter may be corrupted (data race).",
        "Remove the destructor print entirely. GPU memory management should be silent in production code. If you want to debug memory lifecycle, use cuda-memcheck or add a compile-time flag like #ifdef PARALLELNET_DEBUG_MEMORY.",
        [
            "Remove the destructor print from GpuBuffer in kernel_utils.cuh.",
            "Replace the destructor body (after the Issue 5 fix above) with:",
            "",
            "  ~GpuBuffer() noexcept {",
            "      if (data_ != nullptr) {",
            "#ifdef PARALLELNET_DEBUG_MEMORY",
            "          fprintf(stderr, 'GpuBuffer::free count=%zu\\n', count_);",
            "#endif",
            "          cudaError_t err = cudaFree(data_);",
            "          if (err != cudaSuccess && err != cudaErrorCudartUnloading) {",
            "              fprintf(stderr, 'GpuBuffer cudaFree error: %s\\n',",
            "                      cudaGetErrorString(err));",
            "          }",
            "          data_ = nullptr;",
            "      }",
            "  }",
            "",
            "Remove the static free_count variable entirely.",
        ],
        [
            "pytest tests/test_kernels.py produces no GpuBuffer lines in stdout",
            "Build with -DPARALLELNET_DEBUG_MEMORY=1 and verify the print reappears",
        ]
    ))

    # ISSUE 10
    story.extend(issue_card(
        10, "LOW",
        "Windows CUDA DLL setup runs after cupy is already imported",
        ["benchmarks/kernel_bench.py"],
        "configure_windows_cuda_dlls() is defined near the top of kernel_bench.py but is only called inside main(). However, import cupy as cp appears at module level before main() is ever called. On Windows, cupy's DLL dependencies are resolved at import time — by the time configure_windows_cuda_dlls() adds the CUDA bin directory via os.add_dll_directory(), cupy has already either succeeded or failed to import. The function is effectively a no-op on Windows.",
        "On a Windows machine where CUDA_PATH is set but the CUDA bin directory is not in PATH, import cupy fails with a DLL not found error before main() runs. The configure_windows_cuda_dlls() call in main() never executes. This makes the benchmark harder to run on Windows development machines.",
        "Move the configure_windows_cuda_dlls() call to before the import cupy statement. Use a conditional import pattern: call the function, then import cupy.",
        [
            "Restructure the import block in kernel_bench.py so DLL setup runs first.",
            "",
            "Move the function definition above all other imports and call it immediately:",
            "",
            "import os, sys",
            "from pathlib import Path",
            "",
            "def configure_windows_cuda_dlls():",
            "    # ... (same function body) ...",
            "",
            "configure_windows_cuda_dlls()  # must run before importing cupy",
            "",
            "import cupy as cp  # now imports after DLL paths are set",
            "import matplotlib.pyplot as plt",
            "# ... rest of imports ...",
        ],
        [
            "On Windows: remove CUDA bin from PATH, run the benchmark — should succeed via add_dll_directory instead of PATH",
            "On Linux: no change in behaviour",
        ]
    ))

    story.append(PageBreak())

    # ── SECTION: WHAT IS WORKING WELL ─────────────────────────────────────────
    story += h1("What Is Working Well")
    story += [
        body("These parts of the codebase are well-implemented and do not need changes. Understanding what is right is as important as fixing what is wrong."),
        sp(10),
        data_table(
            ["Component", "What's good about it"],
            [
                ["kernels/kernel_utils.cuh — GpuBuffer<T>", "RAII pattern is correct. Move constructor and move assignment are implemented. Copy constructor is deleted (correct — GPU buffers should not be silently copied). The interface (copy_from_host, copy_to_host, data(), size()) is clean and complete."],
                ["kernels/matmul.cu — tiled kernel", "Shared memory tiling is correctly implemented. Boundary checks for non-power-of-two sizes are correct. __syncthreads() calls are in the right places (after load, before use). The two-kernel design (naive for reference, tiled for production) is exactly right."],
                ["kernels/activations.cu", "softmax_forward uses the numerically stable max-subtract pattern. Parallel reduction for max and sum using shared memory is correctly implemented with the right stride halving loop. cross_entropy_loss uses atomicAdd correctly to accumulate partial sums from multiple blocks."],
                ["mpi/gradient_sync.hpp", "MPI_Initialized check before MPI_Init is the correct pattern (prevents double-init if called from mpi4py context). MPI_Barrier before Allreduce prevents deadlock. MPI_Finalized check in destructor prevents double-finalize."],
                ["python/data_loader.py", "Per-channel normalisation using training statistics only (not test). NCHW transpose before flattening. Shard logic is clean and correct. Module-level cache prevents double download."],
                ["tests/test_kernels.py — activation tests", "relu_forward, relu_backward, softmax_forward, and cross_entropy_loss all have NumPy/SciPy reference comparisons. The softmax test checks both values and that rows sum to 1. This is the right level of kernel testing."],
                ["tests/test_bridge.py", "Checks loss decreases, checks save/load roundtrip restores accuracy. Both are meaningful correctness signals."],
                ["python/train.py", "MPI rank and world_size correctly obtained from mpi4py. Only rank 0 prints. Throughput (samples/s) is measured and reported. CLI argument handling is clean."],
                ["benchmarks/gtx1650_results.json", "Recording benchmark results in JSON with the GPU label, sizes, and all three series is the right format for overlaying multiple GPU results later."],
            ],
            [2.0*inch, 4.5*inch]
        ),
        PageBreak(),
    ]

    # ── SECTION: AFTER THE FIXES ───────────────────────────────────────────────
    story += h1("What to Do After Fixing — The Right Order")
    story += [
        body("Fix the three critical issues first, in this order, before doing anything else. Each one depends on the previous."),
        sp(10),
        data_table(
            ["Order", "Fix", "Why this order"],
            [
                ["1st", "Issue 5 — GpuBuffer destructor (CUDA_CHECK)", "Fix this before Issue 1 and 2. Rewriting the engine to use CUDA kernels will involve more CUDA calls and more destruction paths. Having a safe destructor first means you will not accidentally call exit() from a destructor during testing."],
                ["2nd", "Issue 9 — Remove destructor prints", "Do this at the same time as Issue 5 — you are already editing kernel_utils.cuh."],
                ["3rd", "Issue 1 — Dense layer GPU kernels", "This is the core fix. Get forward() working first, verify with test_network that loss still starts at ~2.3, then implement backward(). Do not proceed until numerical gradient check passes."],
                ["4th", "Issue 2 — Network GPU kernels", "Depends on Issue 1 being complete. The output layer uses the same matmul kernel as the hidden layer. Verify that the full training loop in test_network.cpp still converges to <1.0 loss."],
                ["5th", "Issue 4 — Fix benchmark to load matmul.cu", "Now that matmul.cu is being used by the engine, fixing the benchmark to test the same code makes the benchmark numbers meaningful. Re-run and save new gtx1650_results.json."],
                ["6th", "Issue 3 — Enable MPI in Python bridge", "Depends on Issues 1 and 2 being complete. Once the engine uses GPU kernels, enabling MPI and verifying gradient sync produces correct convergence is the final milestone."],
                ["7th", "Issue 7 — Root CMakeLists.txt", "Write this after the engine is working end-to-end. You will know which targets need to be built."],
                ["8th", "Issue 6 — Fix SGD test", "Quick fix, do this whenever you have 20 minutes. Does not depend on anything else."],
                ["9th", "Issue 8 — plot_results.py", "Implement this when you start running dissertation experiments. You need working training first."],
                ["10th", "Issue 10 — Windows DLL import order", "Fix this when you work on Windows next. No urgency."],
            ],
            [0.5*inch, 2.1*inch, 3.9*inch]
        ),
        sp(16),
        h2("Target state after all fixes"),
        body("After all 10 issues are resolved, this is what the system should do:"),
        sp(6),
        blt("<b>Training:</b> mpirun -n 2 python python/train.py produces two workers with identical loss at every epoch. GPU utilisation (nvidia-smi dmon) shows non-zero during training."),
        blt("<b>Kernels:</b> pytest tests/test_kernels.py passes silently (no destructor prints). All kernel tests green."),
        blt("<b>Benchmark:</b> python benchmarks/kernel_bench.py loads matmul.cu, tests it, and produces a chart where the tiled kernel line matches what is in gtx1650_results.json within 10%."),
        blt("<b>C++ tests:</b> cmake -B build . && cmake --build build --parallel 4 && ./build/test_network passes all tests."),
        blt("<b>MPI test:</b> mpirun -n 2 ./build/test_mpi prints [2.0, 3.0, 4.0] on both workers."),
        blt("<b>Accuracy:</b> mpirun -n 1 python train.py --epochs 50 achieves >40% accuracy on CIFAR-10 test set."),
        sp(16),
        *callout("For your dissertation:",
            "Issues 1, 2, and 3 being unfixed means none of your current training timing numbers are valid — you are measuring CPU MLP speed, not GPU distributed training. Do not run dissertation benchmark experiments (Experiments 2, 3, 4) until all three critical fixes are complete and verified. Experiment 1 (kernel benchmark) is also invalid until Issue 4 is fixed. Once all fixes are in place, re-run the full benchmark grid on Colab T4 and record fresh numbers.",
            RED, RED_LIGHT),
        PageBreak(),
    ]

    # ── SECTION: WHAT TO LEARN FROM EACH BUG ──────────────────────────────────
    story += h1("What Each Bug Teaches You")
    story += [
        body("Every bug in this list is a real pattern that comes up in FAANG interviews and production systems. Knowing why each one exists and how to detect it is as valuable as fixing it."),
        sp(10),
        data_table(
            ["Bug", "The lesson", "Interview connection"],
            [
                ["Issues 1 & 2 — CPU engine", "GPU programming is not automatic. Allocating GPU memory is not the same as computing on the GPU. The wiring between data structures and compute functions must be explicit.", '"Walk me through how your kernel is called during training" — you need to trace the call path from Python to CUDA thread. Interviewers ask this at Google, Intel, and ARM.'],
                ["Issue 3 — MPI no-op", "Compile-time flags that silently disable features are dangerous. PARALLELNET_NO_MPI makes multi-worker training silently degenerate into independent training with no error, warning, or log.", '"How do you test distributed systems features?" — the answer involves explicit checks that sync actually happened (gradient checksum comparison), not just that the program ran.'],
                ["Issue 4 — benchmark isolation", "Benchmarks must test the exact code path used in production. A benchmark that tests different code gives false confidence. This is the 'testing in production' antipattern at a smaller scale.", '"How did you validate your performance claims?" — you must be able to say the benchmark runs the same kernel binary as the training engine. If they are different, the claim is not validated.'],
                ["Issue 5 — CUDA_CHECK in destructor", "Destructors must not throw or terminate. This is a core C++ rule. In GPU code it becomes: destructors must not call functions that call exit(). Resource cleanup must always complete.", '"What happens if a CUDA kernel fails during training?" — correct answer: the error is logged, resources are freed, the process exits cleanly. Incorrect: the destructor calls exit() during stack unwinding and you lose the error message.'],
                ["Issue 6 — non-chaining test", "Tests must exercise the actual behaviour being claimed. A test that resets state each iteration does not test chaining or accumulation. This is why numerical gradient checks are the gold standard for neural network testing.", '"How do you test GPU kernels for correctness?" — the answer: compare against a reference implementation and test edge cases like multi-block workloads where thread indexing bugs often hide.'],
                ["Issue 10 — import order", "On Windows, DLL resolution happens at import time, not at function call time. Platform-specific initialisation must run before the affected import, not after.", '"What are common portability issues between Linux and Windows in CUDA projects?" — this specific pattern (add_dll_directory before cupy import) is a well-known gotcha that Intel and ARM engineers encounter when porting GPU tools.'],
            ],
            [1.8*inch, 2.4*inch, 2.3*inch]
        ),
    ]

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Built: {path}")
    print(f"Size: {os.path.getsize(path)//1024} KB")

build()