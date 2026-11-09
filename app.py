import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk
import os
import random
import colorsys

# =========================================================
# AIRCANVAS STUDIO
# =========================================================

CAM_W, CAM_H = 960, 720

BG = "#12131B"
SIDEBAR = "#1B1D28"
PANEL = "#222431"
PANEL_LIGHT = "#303344"
BORDER = "#3B3D50"
ACCENT = "#A78BFA"
ACCENT_DARK = "#8065D8"
TEXT = "#F5F3FA"
MUTED = "#AAA8B8"
GREEN = "#63D6A5"
AMBER = "#E9B66C"

os.makedirs("drawings", exist_ok=True)

BRUSH_PRESETS = {
    "Classic Pen":  {"type": "pen", "width": 1.0, "opacity": 1.0},
    "Fine Liner":   {"type": "pen", "width": 0.45, "opacity": 1.0},
    "Pencil":       {"type": "pencil", "width": 0.65, "opacity": 0.72},
    "Soft Sketch":  {"type": "sketch", "width": 0.85, "opacity": 0.65},
    "Marker":       {"type": "marker", "width": 2.0, "opacity": 0.65},
    "Highlighter":  {"type": "highlighter", "width": 3.0, "opacity": 0.28},
    "Calligraphy":  {"type": "calligraphy", "width": 1.5, "opacity": 1.0},
    "Spray":        {"type": "spray", "width": 2.0, "opacity": 0.8},
}

# Quick-access colours. The full Colour Studio contains 144 hues/shades,
# a grayscale strip, and a custom colour picker.
PALETTE = [
    "#171717", "#FFFFFF", "#777777", "#B9B9B9",
    "#E64F65", "#F27D65", "#F4A261", "#F6C85F",
    "#D7DB65", "#8BCB88", "#45B99A", "#43BCCD",
    "#54A5E8", "#6579DA", "#8D70D6", "#B47BD8",
    "#DB83B5", "#F19CB8", "#8C514B", "#B77A54",
    "#C49B65", "#536F57", "#426D83", "#424C83",
    "#604D7D", "#854F77", "#A6A0A9", "#4A4C5A"
]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.70,
    min_tracking_confidence=0.70
)


def hex_to_bgr(value):
    value = value.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


def joint_angle(a, b, c):
    a = np.array([a.x, a.y, a.z], dtype=float)
    b = np.array([b.x, b.y, b.z], dtype=float)
    c = np.array([c.x, c.y, c.z], dtype=float)
    ba, bc = a - b, c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return 0.0
    cosine = np.dot(ba, bc) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class AirCanvas:
    def __init__(self, root):
        self.root = root
        self.root.title("AirCanvas Studio")
        self.root.configure(bg=BG)

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        self.root.geometry(f"{min(sw - 25, 1700)}x{min(sh - 70, 1050)}")
        self.root.minsize(1050, 720)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to open webcam.")
            root.destroy()
            return

        self.canvas = np.full((CAM_H, CAM_W, 3), 255, dtype=np.uint8)
        self.display_mode = "SCREEN"
        self.tool = "DRAW"
        self.brush_name = "Classic Pen"
        self.color_hex = "#171717"
        self.brush_size = 6
        self.opacity = 100

        self.previous_point = None
        self.smoothed_point = None
        self.shape_start = None
        self.shape_end = None
        self.palm_erasing = False
        self.last_palm_center = None

        self.undo_stack = []
        self.redo_stack = []
        self.running = True

        self.build_interface()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_camera()

    # -----------------------------------------------------
    # UI HELPERS
    # -----------------------------------------------------

    def label(self, parent, text, size=9, color=TEXT, bold=False):
        return tk.Label(
            parent, text=text,
            font=("Segoe UI", size, "bold" if bold else "normal"),
            fg=color, bg=parent.cget("bg")
        )

    def button(self, parent, text, command, selected=False):
        btn = tk.Button(
            parent, text=text, command=command,
            font=("Segoe UI", 9, "bold"),
            fg=TEXT, bg=ACCENT_DARK if selected else PANEL_LIGHT,
            activebackground=ACCENT, activeforeground="white",
            relief="flat", bd=0, padx=13, pady=8, cursor="hand2"
        )
        btn.bind("<Enter>", lambda _e: btn.configure(bg=ACCENT))
        btn.bind("<Leave>", lambda _e: self.refresh_styles())
        return btn

    def section_label(self, parent, text):
        tk.Label(
            parent, text=text.upper(),
            font=("Segoe UI", 8, "bold"),
            fg=MUTED, bg=parent.cget("bg")
        ).pack(anchor="w", pady=(10, 6))

    # -----------------------------------------------------
    # MAIN INTERFACE
    # -----------------------------------------------------

    def build_interface(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=22, pady=(12, 9))

        tk.Label(
            header, text="✦", font=("Segoe UI", 29, "bold"),
            fg=ACCENT, bg=BG
        ).pack(side="left", padx=(0, 10))

        title_box = tk.Frame(header, bg=BG)
        title_box.pack(side="left")
        tk.Label(
            title_box, text="AirCanvas Studio",
            font=("Segoe UI", 21, "bold"), fg=TEXT, bg=BG
        ).pack(anchor="w")
        tk.Label(
            title_box, text="A CREATIVE SPACE POWERED BY HAND TRACKING",
            font=("Segoe UI", 8, "bold"), fg=MUTED, bg=BG
        ).pack(anchor="w")

        self.camera_status = tk.Label(
            header, text="●  CAMERA CONNECTED",
            font=("Segoe UI", 9, "bold"), fg=GREEN, bg=BG
        )
        self.camera_status.pack(side="right", padx=8)

        topbar = tk.Frame(self.root, bg=PANEL, padx=12, pady=8)
        topbar.pack(fill="x", padx=20, pady=(0, 8))

        self.screen_btn = self.button(
            topbar, "Screen Drawing", lambda: self.change_mode("SCREEN")
        )
        self.screen_btn.pack(side="left", padx=3)
        self.canvas_btn = self.button(
            topbar, "Canvas Mode", lambda: self.change_mode("CANVAS")
        )
        self.canvas_btn.pack(side="left", padx=3)
        tk.Frame(topbar, bg=BORDER, width=1, height=27).pack(
            side="left", padx=12
        )
        self.button(topbar, "↶  Undo", self.undo).pack(side="left", padx=3)
        self.button(topbar, "↷  Redo", self.redo).pack(side="left", padx=3)
        self.button(topbar, "Clear", self.clear_canvas).pack(side="left", padx=3)
        self.button(topbar, "Save artwork", self.save_artwork).pack(
            side="right", padx=3
        )

        # Brush controls sit in a slim horizontal strip above the workspace.
        self.build_properties()

        # Main workspace: tools on the left, large camera in the centre,
        # and a dedicated vertical colour dock on the right.
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=4)

        self.sidebar = tk.Frame(main, bg=SIDEBAR, width=155)
        self.sidebar.pack(side="left", fill="y", padx=(0, 10))
        self.sidebar.pack_propagate(False)
        self.build_sidebar()

        self.color_sidebar = tk.Frame(main, bg=SIDEBAR, width=185)
        self.color_sidebar.pack(side="right", fill="y", padx=(10, 0))
        self.color_sidebar.pack_propagate(False)
        self.build_color_sidebar()

        workspace = tk.Frame(main, bg=PANEL, padx=9, pady=9)
        workspace.pack(side="left", fill="both", expand=True)

        self.workspace_title = self.label(
            workspace, "SCREEN DRAWING  /  LIVE CAMERA", 10, TEXT, True
        )
        self.workspace_title.pack(pady=(0, 8))

        self.preview_area = tk.Frame(workspace, bg="#0D1019")
        self.preview_area.pack(fill="both", expand=True)

        self.camera_label = tk.Label(
            self.preview_area, bg="#0D1019", bd=0
        )
        self.whiteboard_label = tk.Label(
            self.preview_area, bg="white", bd=0
        )

        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=22, pady=(5, 9))
        self.status = tk.Label(
            footer,
            text="DRAW: index finger straight, all other fingers folded.",
            font=("Segoe UI", 9), fg=GREEN, bg=BG
        )
        self.status.pack(side="left")
        self.tool_status = tk.Label(
            footer, text="CLASSIC PEN  •  #171717",
            font=("Segoe UI", 9, "bold"), fg=ACCENT, bg=BG
        )
        self.tool_status.pack(side="right")

        self.refresh_styles()
        self.change_mode("SCREEN")

    # -----------------------------------------------------
    # VERTICAL TOOLBAR
    # -----------------------------------------------------

    def build_sidebar(self):
        self.section_label(self.sidebar, "Drawing tools")
        self.tool_buttons = {}

        tools = [
            ("DRAW", "✎   Draw"),
            ("ERASER", "▱   Eraser"),
            ("PAUSE", "Ⅱ   Pause"),
            ("LINE", "／   Line"),
            ("RECTANGLE", "▢   Rectangle"),
            ("CIRCLE", "○   Circle"),
        ]

        for tool, text in tools:
            btn = tk.Button(
                self.sidebar, text=text, anchor="w",
                font=("Segoe UI", 9, "bold"),
                fg=TEXT, bg=SIDEBAR,
                activebackground=ACCENT, activeforeground="white",
                relief="flat", bd=0, padx=12, pady=11,
                cursor="hand2",
                command=lambda t=tool: self.select_tool(t)
            )
            btn.pack(fill="x", padx=7, pady=2)
            btn.bind("<Enter>", lambda _e, b=btn: b.configure(bg=PANEL_LIGHT))
            btn.bind("<Leave>", lambda _e: self.refresh_styles())
            self.tool_buttons[tool] = btn

        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(
            fill="x", padx=12, pady=12
        )
        self.label(self.sidebar, "Hand controls", 8, MUTED, True).pack(
            anchor="w", padx=13
        )
        tk.Label(
            self.sidebar,
            text=(
                "DRAW\n"
                "Index finger straight\n"
                "Other 3 fingers folded\n\n"
                "STOP\n"
                "Bend index OR raise\n"
                "any other finger\n\n"
                "PALM ERASER\n"
                "Open your palm\n"
                "at any time to wipe"
            ),
            font=("Segoe UI", 8), fg=MUTED, bg=SIDEBAR,
            justify="left"
        ).pack(anchor="w", padx=13, pady=8)

    # -----------------------------------------------------
    # BRUSH / COLOUR PROPERTIES
    # -----------------------------------------------------

    def build_properties(self):
        properties = tk.Frame(self.root, bg=PANEL, padx=14, pady=5)
        properties.pack(fill="x", padx=20, pady=(0, 7))

        brush_box = tk.Frame(properties, bg=PANEL)
        brush_box.pack(side="left", padx=(0, 18))
        self.section_label(brush_box, "Brush style")

        self.brush_var = tk.StringVar(value=self.brush_name)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Air.TCombobox",
            fieldbackground=PANEL_LIGHT, background=PANEL_LIGHT,
            foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER,
            lightcolor=BORDER, darkcolor=BORDER, padding=5
        )
        self.brush_combo = ttk.Combobox(
            brush_box, textvariable=self.brush_var,
            values=list(BRUSH_PRESETS.keys()),
            state="readonly", width=17, style="Air.TCombobox"
        )
        self.brush_combo.pack(anchor="w")
        self.brush_combo.bind("<<ComboboxSelected>>", self.change_brush)

        size_box = tk.Frame(properties, bg=PANEL)
        size_box.pack(side="left", padx=18)
        self.section_label(size_box, "Brush size")
        self.size_value = tk.Label(
            size_box, text="6 px", font=("Segoe UI", 9, "bold"),
            fg=TEXT, bg=PANEL
        )
        self.size_value.pack(anchor="w")
        self.size_slider = tk.Scale(
            size_box, from_=1, to=50, orient="horizontal", length=145,
            bg=PANEL, fg=TEXT, troughcolor=PANEL_LIGHT,
            activebackground=ACCENT, highlightthickness=0, bd=0,
            showvalue=False, command=self.change_size
        )
        self.size_slider.set(6)
        self.size_slider.pack()

        opacity_box = tk.Frame(properties, bg=PANEL)
        opacity_box.pack(side="left", padx=18)
        self.section_label(opacity_box, "Opacity")
        self.opacity_value = tk.Label(
            opacity_box, text="100%", font=("Segoe UI", 9, "bold"),
            fg=TEXT, bg=PANEL
        )
        self.opacity_value.pack(anchor="w")
        self.opacity_slider = tk.Scale(
            opacity_box, from_=10, to=100, orient="horizontal", length=130,
            bg=PANEL, fg=TEXT, troughcolor=PANEL_LIGHT,
            activebackground=ACCENT, highlightthickness=0, bd=0,
            showvalue=False, command=self.change_opacity
        )
        self.opacity_slider.set(100)
        self.opacity_slider.pack()

    def build_color_sidebar(self):
        self.section_label(self.color_sidebar, "Colour palette")

        selected_card = tk.Frame(
            self.color_sidebar, bg=PANEL_LIGHT, padx=9, pady=9
        )
        selected_card.pack(fill="x", padx=8, pady=(2, 9))

        self.selected_color_preview = tk.Label(
            selected_card, text=" ", bg=self.color_hex,
            width=5, height=2, relief="flat"
        )
        self.selected_color_preview.pack(side="left", padx=(0, 8))

        selected_text = tk.Frame(selected_card, bg=PANEL_LIGHT)
        selected_text.pack(side="left", fill="x", expand=True)
        tk.Label(
            selected_text, text="SELECTED",
            font=("Segoe UI", 7, "bold"), fg=MUTED, bg=PANEL_LIGHT
        ).pack(anchor="w")
        self.selected_color_code = tk.Label(
            selected_text, text=self.color_hex.upper(),
            font=("Segoe UI", 9, "bold"), fg=TEXT, bg=PANEL_LIGHT
        )
        self.selected_color_code.pack(anchor="w")

        tk.Label(
            self.color_sidebar, text="QUICK COLOURS",
            font=("Segoe UI", 8, "bold"), fg=MUTED, bg=SIDEBAR
        ).pack(anchor="w", padx=12, pady=(3, 6))

        self.palette_buttons = {}
        swatch_grid = tk.Frame(self.color_sidebar, bg=SIDEBAR)
        swatch_grid.pack(fill="x", padx=9)

        # Four columns make the quick palette readable in the vertical dock.
        for i, color in enumerate(PALETTE):
            btn = tk.Button(
                swatch_grid, bg=color, activebackground=color,
                width=3, height=1, relief="flat", bd=0,
                cursor="hand2", highlightthickness=1,
                highlightbackground=BORDER,
                command=lambda c=color: self.choose_color(c)
            )
            btn.grid(row=i // 4, column=i % 4, padx=3, pady=4, sticky="nsew")
            self.palette_buttons[color] = btn

        tk.Frame(self.color_sidebar, bg=BORDER, height=1).pack(
            fill="x", padx=12, pady=11
        )
        tk.Button(
            self.color_sidebar, text="＋  Colour Studio",
            command=self.open_full_palette,
            font=("Segoe UI", 9, "bold"), fg="white",
            bg=ACCENT_DARK, activebackground=ACCENT,
            relief="flat", bd=0, padx=8, pady=10,
            cursor="hand2"
        ).pack(fill="x", padx=9)

        tk.Button(
            self.color_sidebar, text="Custom colour…",
            command=self.custom_color,
            font=("Segoe UI", 8, "bold"), fg=TEXT,
            bg=PANEL_LIGHT, activebackground=ACCENT,
            relief="flat", bd=0, padx=8, pady=8,
            cursor="hand2"
        ).pack(fill="x", padx=9, pady=(7, 0))

    # -----------------------------------------------------
    # FULL COLOUR STUDIO
    # -----------------------------------------------------

    def open_full_palette(self):
        win = tk.Toplevel(self.root)
        win.title("AirCanvas | Colour Studio")
        win.configure(bg=BG)
        win.geometry("820x650")
        win.minsize(760, 580)
        win.transient(self.root)

        tk.Label(
            win, text="COLOUR STUDIO",
            font=("Segoe UI", 19, "bold"), fg=TEXT, bg=BG
        ).pack(anchor="w", padx=22, pady=(16, 2))
        tk.Label(
            win, text="1. Choose a colour family   2. Pick a shade   3. Apply",
            font=("Segoe UI", 9), fg=MUTED, bg=BG
        ).pack(anchor="w", padx=22, pady=(0, 12))

        body = tk.Frame(win, bg=PANEL, padx=14, pady=12)
        body.pack(fill="both", expand=True, padx=18, pady=4)

        family_names = [
            "Red", "Red-orange", "Orange", "Amber", "Yellow", "Lime",
            "Green", "Teal", "Cyan", "Blue", "Indigo", "Violet",
            "Purple", "Magenta", "Pink", "Rose"
        ]
        shade_levels = [
            (0.95, 1.00), (0.82, 1.00), (0.68, 0.96), (0.55, 0.88),
            (0.72, 0.76), (0.88, 0.64), (0.48, 0.60), (0.30, 0.48)
        ]

        self.palette_choice = tk.StringVar(value=self.color_hex)
        self.palette_choice_label = tk.Label(
            body, text=f"Selected: {self.color_hex.upper()}",
            font=("Segoe UI", 9, "bold"), fg=TEXT, bg=PANEL
        )
        self.palette_choice_label.grid(
            row=0, column=0, columnspan=16, sticky="w", pady=(0, 8)
        )

        for hue_index, family in enumerate(family_names):
            hue = hue_index / len(family_names)
            tk.Label(
                body, text=family, font=("Segoe UI", 7, "bold"),
                fg=MUTED, bg=PANEL
            ).grid(row=1, column=hue_index, padx=2, pady=(0, 5))
            for shade_index, (sat, val) in enumerate(shade_levels):
                r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
                color = "#{:02X}{:02X}{:02X}".format(
                    round(r * 255), round(g * 255), round(b * 255)
                )
                tk.Button(
                    body, bg=color, activebackground=color,
                    width=3, height=1, relief="flat", bd=0,
                    cursor="hand2",
                    command=lambda c=color: self.set_palette_preview(c)
                ).grid(
                    row=shade_index + 2, column=hue_index,
                    padx=2, pady=3, sticky="nsew"
                )

        tk.Label(
            body, text="NEUTRALS", font=("Segoe UI", 8, "bold"),
            fg=MUTED, bg=PANEL
        ).grid(row=11, column=0, columnspan=16, sticky="w", pady=(12, 5))

        neutral_row = tk.Frame(body, bg=PANEL)
        neutral_row.grid(row=12, column=0, columnspan=16, sticky="w")
        for value in range(0, 256, 16):
            shade = "#{:02X}{:02X}{:02X}".format(value, value, value)
            tk.Button(
                neutral_row, bg=shade, activebackground=shade,
                width=3, height=1, relief="flat", bd=0,
                cursor="hand2",
                command=lambda c=shade: self.set_palette_preview(c)
            ).pack(side="left", padx=2)

        actions = tk.Frame(win, bg=BG)
        actions.pack(fill="x", padx=20, pady=12)
        tk.Button(
            actions, text="Custom colour picker",
            command=lambda: self.custom_color(win),
            font=("Segoe UI", 9, "bold"), fg="white",
            bg=PANEL_LIGHT, activebackground=ACCENT,
            relief="flat", padx=13, pady=9
        ).pack(side="left")
        tk.Button(
            actions, text="Apply selected colour",
            command=lambda: self.apply_palette_choice(win),
            font=("Segoe UI", 9, "bold"), fg="white",
            bg=ACCENT_DARK, activebackground=ACCENT,
            relief="flat", padx=15, pady=9
        ).pack(side="right")

    def set_palette_preview(self, color):
        self.palette_choice.set(color)
        self.palette_choice_label.configure(
            text=f"Selected: {color.upper()}"
        )

    def apply_palette_choice(self, window):
        self.choose_color(self.palette_choice.get())
        window.destroy()

    def set_colour_and_close(self, color, window):
        self.choose_color(color)
        window.destroy()

    def choose_color(self, color):
        self.color_hex = color
        if hasattr(self, "selected_color_preview"):
            self.selected_color_preview.configure(bg=color)
        if hasattr(self, "selected_color_code"):
            self.selected_color_code.configure(text=color.upper())
        self.tool_status.configure(
            text=f"{self.brush_name.upper()}  •  {color.upper()}"
        )

    def custom_color(self, parent=None):
        result = colorchooser.askcolor(
            title="Choose a brush colour", parent=parent or self.root
        )
        if result and result[1]:
            self.choose_color(result[1])

    # -----------------------------------------------------
    # STYLE / MODES / TOOLS
    # -----------------------------------------------------

    def refresh_styles(self):
        for name, btn in getattr(self, "tool_buttons", {}).items():
            btn.configure(
                bg=ACCENT_DARK if name == self.tool else SIDEBAR,
                fg="white" if name == self.tool else TEXT
            )
        if hasattr(self, "screen_btn"):
            self.screen_btn.configure(
                bg=ACCENT_DARK if self.display_mode == "SCREEN" else PANEL_LIGHT
            )
            self.canvas_btn.configure(
                bg=ACCENT_DARK if self.display_mode == "CANVAS" else PANEL_LIGHT
            )

    def change_mode(self, mode):
        self.display_mode = mode
        self.previous_point = None
        self.shape_start = self.shape_end = None

        self.camera_label.pack_forget()
        self.whiteboard_label.pack_forget()

        if mode == "SCREEN":
            self.workspace_title.configure(
                text="SCREEN DRAWING  /  LIVE CAMERA"
            )
            self.camera_label.pack(fill="both", expand=True)
        else:
            self.workspace_title.configure(
                text="CANVAS MODE  /  LIVE CAMERA + WHITEBOARD"
            )
            # Two equal expanding panels. The camera is not cropped.
            self.camera_label.pack(
                side="left", fill="both", expand=True, padx=(0, 4)
            )
            self.whiteboard_label.pack(
                side="left", fill="both", expand=True, padx=(4, 0)
            )
        self.refresh_styles()

    def select_tool(self, tool):
        self.tool = tool
        self.previous_point = None
        self.shape_start = self.shape_end = None
        self.palm_erasing = False
        self.last_palm_center = None

        self.tool_status.configure(
            text=f"{tool}  •  {self.color_hex.upper()}"
        )
        messages = {
            "DRAW": "Draw: keep index straight and fold the other three fingers.",
            "ERASER": "Eraser selected. Open your palm to wipe (works with any tool).",
            "PAUSE": "Drawing paused. Select Draw to continue.",
            "LINE": "Line: hold the index-only gesture to set the line, then relax to finish.",
            "RECTANGLE": "Rectangle: hold the index-only gesture to set the shape, then relax.",
            "CIRCLE": "Circle: hold the index-only gesture to set the circle, then relax."
        }
        self.status.configure(text=messages.get(tool, tool), fg=GREEN)
        self.refresh_styles()

    def change_brush(self, _event=None):
        self.brush_name = self.brush_var.get()
        self.tool_status.configure(
            text=f"{self.brush_name.upper()}  •  {self.color_hex.upper()}"
        )

    def change_size(self, value):
        self.brush_size = int(float(value))
        self.size_value.configure(text=f"{self.brush_size} px")

    def change_opacity(self, value):
        self.opacity = int(float(value))
        self.opacity_value.configure(text=f"{self.opacity}%")

    # -----------------------------------------------------
    # HAND GESTURE DETECTION
    # -----------------------------------------------------

    def finger_angles(self, lm):
        return {
            "index": joint_angle(lm[5], lm[6], lm[7]),
            "middle": joint_angle(lm[9], lm[10], lm[11]),
            "ring": joint_angle(lm[13], lm[14], lm[15]),
            "pinky": joint_angle(lm[17], lm[18], lm[19])
        }

    def only_index_active(self, lm):
        """
        Strict draw pose:
        - index finger must be straight
        - middle, ring and pinky must be folded
        - thumb is ignored so it can rest naturally
        The strict folded-finger limit makes raising another finger stop drawing.
        """
        angles = self.finger_angles(lm)
        return (
            angles["index"] >= 158
            and angles["middle"] <= 142
            and angles["ring"] <= 142
            and angles["pinky"] <= 142
        )

    def is_open_palm(self, lm):
        angles = self.finger_angles(lm)
        return all(angles[name] >= 150 for name in (
            "index", "middle", "ring", "pinky"
        ))

    def get_point(self, lm):
        x = int(np.clip(lm[8].x * CAM_W, 0, CAM_W - 1))
        y = int(np.clip(lm[8].y * CAM_H, 0, CAM_H - 1))

        if self.smoothed_point is None:
            self.smoothed_point = (x, y)
        else:
            old_x, old_y = self.smoothed_point
            # Responsive smoothing: less lag while reducing small camera jitters.
            self.smoothed_point = (
                int(0.30 * old_x + 0.70 * x),
                int(0.30 * old_y + 0.70 * y)
            )
        return self.smoothed_point

    def get_palm_area(self, lm):
        # Centre of palm, excluding fingertips. Radius scales with hand size.
        palm_ids = [0, 5, 9, 13, 17]
        cx = int(np.mean([lm[i].x for i in palm_ids]) * CAM_W)
        cy = int(np.mean([lm[i].y for i in palm_ids]) * CAM_H)
        center = (
            int(np.clip(cx, 0, CAM_W - 1)),
            int(np.clip(cy, 0, CAM_H - 1))
        )
        palm_width = np.linalg.norm(
            np.array([lm[5].x, lm[5].y]) -
            np.array([lm[17].x, lm[17].y])
        ) * CAM_W
        radius = int(np.clip(palm_width * 1.45, 55, 180))
        return center, radius

    # -----------------------------------------------------
    # DRAWING ENGINE
    # -----------------------------------------------------

    def save_undo(self):
        self.undo_stack.append(self.canvas.copy())
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def blend_line(self, image, start, end, color, thickness, alpha):
        overlay = image.copy()
        cv2.line(
            overlay, start, end, color,
            max(1, int(thickness)), cv2.LINE_AA
        )
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)

    def draw_brush(self, image, start, end):
        preset = BRUSH_PRESETS[self.brush_name]
        kind = preset["type"]
        color = hex_to_bgr(self.color_hex)
        width = max(1, int(self.brush_size * preset["width"]))
        alpha = min(1.0, self.opacity / 100 * preset["opacity"])

        if kind == "pen":
            self.blend_line(image, start, end, color, width, alpha)
        elif kind == "pencil":
            self.blend_line(image, start, end, (95, 95, 95), width, alpha)
        elif kind == "sketch":
            self.blend_line(image, start, end, color, width, alpha)
            dx, dy = random.randint(-2, 2), random.randint(-2, 2)
            cv2.line(
                image, (start[0] + dx, start[1] + dy),
                (end[0] + dx, end[1] + dy), color, 1, cv2.LINE_AA
            )
        elif kind == "marker":
            self.blend_line(image, start, end, color, width * 2, alpha)
        elif kind == "highlighter":
            self.blend_line(
                image, start, end, color, width * 3, min(0.30, alpha)
            )
        elif kind == "calligraphy":
            thickness = width if abs(end[0]-start[0]) > abs(end[1]-start[1]) else width * 2
            self.blend_line(image, start, end, color, thickness, alpha)
        elif kind == "spray":
            distance = int(np.linalg.norm(np.array(end) - np.array(start)))
            steps = max(1, distance // 5)
            for i in range(steps + 1):
                t = i / steps
                px = int(start[0] + (end[0] - start[0]) * t)
                py = int(start[1] + (end[1] - start[1]) * t)
                for _ in range(10):
                    sx = random.randint(-width * 2, width * 2)
                    sy = random.randint(-width * 2, width * 2)
                    if sx*sx + sy*sy <= (width * 2) ** 2:
                        cv2.circle(image, (px + sx, py + sy), 1, color, -1)

    def process_drawing(self, point, active):
        if self.tool == "PAUSE":
            self.previous_point = None
            self.shape_start = self.shape_end = None
            return

        if self.tool in ("LINE", "RECTANGLE", "CIRCLE"):
            if active:
                if self.shape_start is None:
                    self.save_undo()
                    self.shape_start = point
                self.shape_end = point
            elif self.shape_start is not None:
                self.finish_shape()
            return

        if self.tool != "DRAW":
            self.previous_point = None
            return

        # Any slight bend of index or extension of another finger stops the stroke.
        if not active:
            self.previous_point = None
            return

        if self.previous_point is None:
            self.save_undo()
            self.previous_point = point
            return

        self.draw_brush(self.canvas, self.previous_point, point)
        self.previous_point = point

    def erase_with_palm(self, lm):
        center, radius = self.get_palm_area(lm)
        if not self.palm_erasing:
            self.save_undo()
            self.palm_erasing = True

        # White, filled, anti-aliased disc behaves like wiping a roughing board.
        if self.last_palm_center is not None:
            cv2.line(
                self.canvas, self.last_palm_center, center,
                (255, 255, 255), radius * 2, cv2.LINE_AA
            )
        cv2.circle(
            self.canvas, center, radius,
            (255, 255, 255), -1, cv2.LINE_AA
        )
        self.last_palm_center = center
        return center, radius

    def finish_shape(self):
        if self.shape_start is None:
            return
        start = self.shape_start
        end = self.shape_end or start
        color = hex_to_bgr(self.color_hex)

        if self.tool == "LINE":
            cv2.line(self.canvas, start, end, color, self.brush_size, cv2.LINE_AA)
        elif self.tool == "RECTANGLE":
            cv2.rectangle(self.canvas, start, end, color, self.brush_size, cv2.LINE_AA)
        elif self.tool == "CIRCLE":
            radius = int(np.linalg.norm(np.array(end) - np.array(start)))
            cv2.circle(self.canvas, start, radius, color, self.brush_size, cv2.LINE_AA)

        self.shape_start = self.shape_end = None

    def canvas_preview(self):
        preview = self.canvas.copy()
        if self.shape_start is None or self.shape_end is None:
            return preview
        start, end = self.shape_start, self.shape_end
        color = hex_to_bgr(self.color_hex)

        if self.tool == "LINE":
            cv2.line(preview, start, end, color, self.brush_size, cv2.LINE_AA)
        elif self.tool == "RECTANGLE":
            cv2.rectangle(preview, start, end, color, self.brush_size, cv2.LINE_AA)
        elif self.tool == "CIRCLE":
            radius = int(np.linalg.norm(np.array(end) - np.array(start)))
            cv2.circle(preview, start, radius, color, self.brush_size, cv2.LINE_AA)
        return preview

    # -----------------------------------------------------
    # UNDO / REDO / CLEAR / SAVE
    # -----------------------------------------------------

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.canvas.copy())
            self.canvas = self.undo_stack.pop()
            self.previous_point = None
            self.palm_erasing = False

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.canvas.copy())
            self.canvas = self.redo_stack.pop()
            self.previous_point = None
            self.palm_erasing = False

    def clear_canvas(self):
        if np.any(self.canvas != 255):
            self.save_undo()
            self.canvas[:] = 255
            self.previous_point = None
            self.palm_erasing = False

    def save_artwork(self):
        path = filedialog.asksaveasfilename(
            initialdir="drawings", defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
        )
        if path:
            cv2.imwrite(path, self.canvas)
            messagebox.showinfo("Saved", "Your artwork has been saved successfully.")

    # -----------------------------------------------------
    # DISPLAY / CAMERA
    # -----------------------------------------------------

    def display_image(self, label, frame):
        # Fit the complete image inside the available panel, preserving aspect ratio.
        width = max(1, label.winfo_width())
        height = max(1, label.winfo_height())
        if width < 50 or height < 50:
            width, height = CAM_W, CAM_H

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo

    def update_camera(self):
        """Read the webcam, detect the hand and update the drawing surface.

        Gesture priority:
        1. PAUSE disables all hand actions.
        2. A fully open palm automatically erases, regardless of the selected
           drawing tool. No Eraser toolbar click is required.
        3. Index-only pose draws when Draw or a shape tool is selected.
        4. Any other pose stops the current stroke.
        """
        if not self.running:
            return

        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (CAM_W, CAM_H))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                lm = hand.landmark
                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                if self.tool == "PAUSE":
                    self.previous_point = None
                    self.shape_start = self.shape_end = None
                    self.palm_erasing = False
                    self.last_palm_center = None
                    self.status.configure(
                        text="Paused • hand gestures are temporarily disabled",
                        fg=AMBER
                    )

                elif self.is_open_palm(lm):
                    # Automatic palm eraser. This branch runs even when DRAW,
                    # LINE, RECTANGLE or CIRCLE is selected.
                    self.previous_point = None
                    self.shape_start = self.shape_end = None
                    center, radius = self.erase_with_palm(lm)
                    cv2.circle(frame, center, radius, (0, 120, 255), 2)
                    cv2.putText(
                        frame, "AUTO PALM ERASER",
                        (20, CAM_H - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 120, 255), 2, cv2.LINE_AA
                    )
                    self.status.configure(
                        text="Auto erasing • move your open palm over the drawing",
                        fg=AMBER
                    )

                else:
                    # Leaving the open-palm pose ends the current erase pass.
                    self.palm_erasing = False
                    self.last_palm_center = None
                    point = self.get_point(lm)
                    active = self.only_index_active(lm)

                    if self.tool == "ERASER":
                        # Eraser tool is a manual eraser mode, but the broad
                        # open-palm wipe above remains automatic in every mode.
                        self.previous_point = None
                        self.status.configure(
                            text="Palm eraser selected • open your whole hand to wipe",
                            fg=AMBER
                        )
                    else:
                        self.process_drawing(point, active)
                        cursor_color = (0, 255, 100) if active else (0, 190, 255)
                        cv2.circle(frame, point, 9, cursor_color, -1)

                        if active:
                            self.status.configure(
                                text="Drawing active • index finger only",
                                fg=GREEN
                            )
                        else:
                            self.status.configure(
                                text="Stopped • bend the index finger or raise another finger",
                                fg=AMBER
                            )

            else:
                # Reset stroke state when the hand leaves the camera view.
                self.previous_point = None
                self.smoothed_point = None
                self.shape_start = self.shape_end = None
                self.palm_erasing = False
                self.last_palm_center = None
                self.status.configure(
                    text="Show one hand in the camera to begin.",
                    fg=MUTED
                )

            preview = self.canvas_preview()
            if self.display_mode == "SCREEN":
                # Overlay only non-white drawing pixels onto the live camera.
                screen = frame.copy()
                gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                mask = cv2.threshold(
                    gray, 245, 255, cv2.THRESH_BINARY_INV
                )[1]
                screen[mask > 0] = self.canvas[mask > 0]
                self.display_image(self.camera_label, screen)
            else:
                self.display_image(self.camera_label, frame)
                self.display_image(self.whiteboard_label, preview)

        else:
            self.camera_status.configure(
                text="●  CAMERA DISCONNECTED", fg="#F07878"
            )

        self.root.after(15, self.update_camera)

    def close(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        hands.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AirCanvas(root)
    root.mainloop()
