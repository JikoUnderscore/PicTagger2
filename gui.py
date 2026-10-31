import dataclasses
import os
import pathlib
import subprocess
import tempfile
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import typing
from tkinter import ttk

import exiftool
import numpy as np
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
import PIL.ImageTk
import sklearn.cluster

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


class ScrollableFrame(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Widget,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.content_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.content_frame, anchor="nw"
        )

        # Event bindings
        _ = self.content_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        _ = self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(
                self.canvas_window,
                width=e.width,
            ),
        )

        # Mouse wheel scrolling
        _ = self.content_frame.bind(
            "<MouseWheel>", lambda e: self.canvas.yview_scroll(-e.delta // 60, "units")
        )
        _ = self.canvas.bind(
            "<MouseWheel>", lambda e: self.canvas.yview_scroll(-e.delta // 60, "units")
        )


@dataclasses.dataclass(slots=True)
class PersonFaces:
    path: list[str]
    face_box: list[np.ndarray]


@dataclasses.dataclass(slots=True)
class ImageElentData:
    data: PIL.ImageTk.PhotoImage
    label: ttk.Label
    listbox: tk.Listbox
    is_writen: bool
    person_faces_path: str


@dataclasses.dataclass(slots=True)
class View:
    index: int
    frame: typing.Any
    elements: list[ImageElentData]


def resize_image(image: PIL.Image.Image, max_w: int, max_h: int) -> PIL.Image.Image:
    w, h = image.size
    if w <= max_w and h <= max_h:
        return image
    ratio = min(max_w / w, max_h / h)
    return image.resize((int(w * ratio), int(h * ratio)), PIL.Image.Resampling.LANCZOS)


def read_tags(filepath: str, et: exiftool.ExifToolHelper) -> list[str]:
    def repair_cp1251_misdecode(value: str) -> str:
        try:
            return value.encode("cp1251").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value

    metadata: list[dict[str, str | list[str]]] = et.get_tags(
        filepath, ["IPTC:Keywords", "XMP:Subject", "EXIF:XPKeywords"]
    )

    if not metadata:
        return []

    meta = metadata[0]
    collected: list[str] = []

    # ---- IPTC
    if "IPTC:Keywords" in meta:
        value = meta["IPTC:Keywords"]
        if isinstance(value, list):
            collected.extend(value)
        elif isinstance(value, str):
            collected.append(value)

    # ---- XMP
    if "XMP:Subject" in meta:
        value = meta["XMP:Subject"]
        if isinstance(value, list):
            collected.extend(value)
        elif isinstance(value, str):
            collected.append(value)

    # ---- EXIF XPKeywords (semicolon separated)
    if "EXIF:XPKeywords" in meta:
        value = meta["EXIF:XPKeywords"]
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(";") if p.strip()]
            collected.extend(parts)

    # ---- Apply cp1251 repair
    repaired = [repair_cp1251_misdecode(tag) for tag in collected]

    # ---- Deduplicate while preserving order
    seen: set[str] = set()
    final: list[str] = []
    for tag in repaired:
        if tag not in seen:
            seen.add(tag)
            final.append(tag)

    return final


@typing.final
class TagEditorApp:
    def __init__(
        self, root: tk.Tk, et: exiftool.ExifToolHelper, selected_npz_file: str
    ):
        self.root = root
        root.geometry("1280x720")
        root.title("Tag Editor")
        self.et = et
        self.defaultbg = root.cget("bg")
        self.curent_selected_listbox: tk.Listbox | None = None

        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=0)
        root.columnconfigure(2, weight=1)

        # Tag Editor Panel (Column 0)
        self.controls_tag_editor = ttk.Frame(root)
        self.controls_tag_editor.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        cr = 0

        self.entry_var = tk.StringVar()
        self.entry = ttk.Entry(self.controls_tag_editor, textvariable=self.entry_var)
        self.entry.grid(row=cr, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.add_button = ttk.Button(
            self.controls_tag_editor, text="Add to selected", command=self.add_tag
        )

        self.add_button.grid(
            row=(cr := cr + 1), column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )

        self.addall_button = ttk.Button(
            self.controls_tag_editor, text="Add to all", command=self.addall_tag
        )
        self.addall_button.grid(
            row=(cr := cr + 1), column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )

        self.remove_button = ttk.Button(
            self.controls_tag_editor,
            text="Remove Selected",
            command=self.remove_selected,
        )
        self.remove_button.grid(
            row=(cr := cr + 1), column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )

        self.apply_button = ttk.Button(
            self.controls_tag_editor,
            text="Write all(F1)",
            command=self.write_all_tags,
        )
        self.apply_button.grid(
            row=(cr := cr + 1), column=0, columnspan=2, sticky="ew", pady=(10, 10)
        )

        self.prev_button = ttk.Button(
            self.controls_tag_editor, text="<<", command=self.prev_person
        )
        self.prev_button.grid(row=(cr := cr + 1), column=0, sticky="ew")

        self.next_button = ttk.Button(
            self.controls_tag_editor, text=">>", command=self.next_person
        )
        self.next_button.grid(row=cr, column=1, sticky="ew")

        self.skip_enabled = tk.BooleanVar(value=False)
        self.skip_is_written_check = ttk.Checkbutton(self.controls_tag_editor, text="Skip the check on << and >> buttons", variable=self.skip_enabled)
        self.skip_is_written_check.grid(row=(cr := cr + 1), column=0, columnspan=2, sticky="ew")

        # Recent Tags Panel (Column 1)
        self.recent_tags: list[str] = []
        self.recent_frame = ttk.Frame(self.root)
        self.recent_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.recent_label = ttk.Label(self.recent_frame, text="Recent Tags")
        self.recent_label.grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.recent_buttons_container = ttk.Frame(self.recent_frame)
        self.recent_buttons_container.grid(row=1, column=0, sticky="nsew")

        # Image Panel (Column 2 - expands to fill window)
        self.face_images = ttk.Frame(self.root)
        self.face_images.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.face_images.rowconfigure(0, weight=1)
        self.face_images.columnconfigure(0, weight=1)

        self.persons: list[PersonFaces] = []
        self.cache_view_of_person_faces: list[View] = []
        self.index = 0

        self.parce_npz_file(selected_npz_file)
        self.show_person()

    def add_tag(self):
        tag = self.entry_var.get().strip()
        if not tag:
            return

        if self.curent_selected_listbox is not None:
            self.curent_selected_listbox.insert("end", tag)
            self.entry_var.set("")
            self._add_to_recent(tag)

    def addall_tag(self):
        tag = self.entry_var.get().strip()
        if not tag:
            return

        for ele in self.cache_view_of_person_faces[self.index].elements:
            ele.listbox.insert("end", tag)
            self.entry_var.set("")

        self._add_to_recent(tag)

    def _add_to_recent(self, tag: str):
        if tag in self.recent_tags:
            return

        self.recent_tags.insert(0, tag)

        def add_from_recent(t: str):
            if self.curent_selected_listbox is not None:
                _ = (self.curent_selected_listbox.insert("end", t),)

        button = ttk.Button(
            self.recent_buttons_container,
            text=tag,
            command=lambda: add_from_recent(tag),
        )

        button.grid(row=len(self.recent_tags), column=0, sticky="ew", pady=2)

    def remove_selected(self):
        if self.curent_selected_listbox is not None:
            selection = self.curent_selected_listbox.curselection()
            if selection:
                self.curent_selected_listbox.config(background=self.defaultbg)
                self.curent_selected_listbox.delete(selection)
                self.curent_selected_listbox = None
        self.root.focus_force()

    def _is_written_for_all_elements_warning(self) -> bool:
        if self.skip_enabled.get():
            self.skip_enabled.set(False)
            return True

        is_writen = True
        for ele in self.cache_view_of_person_faces[self.index].elements:
            if ele.is_writen == False:
                tkinter.messagebox.showinfo(
                    "Non writen data", "Write all data before proceeding, OR check the checkbox"
                )
                is_writen = False
                break
        return is_writen

    def next_person(self):
        if self._is_written_for_all_elements_warning() == False:
            return

        if self.curent_selected_listbox is not None:
            self.curent_selected_listbox.config(background=self.defaultbg)
        self.curent_selected_listbox = None
        if self.index < len(self.persons) - 1:
            self.index += 1
            self.show_person()

    def prev_person(self):
        if self._is_written_for_all_elements_warning() == False:
                return
    
        if self.curent_selected_listbox is not None:
            self.curent_selected_listbox.config(background=self.defaultbg)
        self.curent_selected_listbox = None
        if self.index > 0:
            self.index -= 1
            self.show_person()

    def write_all_tags(self):
        for ele in self.cache_view_of_person_faces[self.index].elements:
            filepath = ele.person_faces_path
            args_path: str

            with tempfile.NamedTemporaryFile(
                mode="wt", encoding="utf-8", delete=False
            ) as argsf:
                argsf.write("-overwrite_original\n")
                argsf.write("-P\n")
                argsf.write("-charset\nutf8\n")
                argsf.write("-keywords=\n")  # clear existing
                tags: tuple[str, ...] = ele.listbox.get(0, "end")
                for tag in tags:
                    if tag:
                        _ = argsf.write(f"-keywords={tag}\n")
                args_path = argsf.name

            cmd = ["exiftool", "-@", args_path, filepath]
            try:
                print(cmd)
                output = subprocess.run(cmd, capture_output=True, check=True, text=True)
                print(f"{output=}")
                ele.is_writen = True
            except subprocess.CalledProcessError as e:
                details = f"stdout:\n{e.stdout}\nstderr:\n{e.stderr}"
                _ = tkinter.messagebox.showerror(
                    "Error on running command",
                    f"cmd: {cmd}\nCommand failed with exit code {e.returncode}\n{details}",
                )
                print(
                    f"cmd: {cmd}\nCommand failed with exit code {e.returncode}\n{details}"
                )
            finally:
                os.unlink(args_path)
        tkinter.messagebox.showinfo("Done", "work is doen")
        # os.remove(args_file)
        # # self.write_tags(filepath, self.current_tags)
        # # print(f"Done {filepath}")

    def show_person(self):
        for view in self.cache_view_of_person_faces:
            if self.index == view.index:
                view.frame.tkraise()
                return

        person = self.persons[self.index]

        pview = ttk.Frame(self.face_images)
        pview.grid(row=0, column=0, sticky="nsew")
        pview.rowconfigure(0, weight=1)
        pview.columnconfigure(0, weight=1)
        scroll = ScrollableFrame(pview)
        scroll.pack(fill="both", expand=True)

        view = scroll.content_frame

        # _ = view.bind(
        #     "<MouseWheel>", lambda e: scroll.canvas.yview_scroll(-e.delta // 60, "units")
        # )
        # _ = pview.bind(
        #     "<MouseWheel>", lambda e: scroll.canvas.yview_scroll(-e.delta // 60, "units")
        # )
        # _ = scroll.canvas.bind(
        #     "<MouseWheel>", lambda e: scroll.canvas.yview_scroll(-e.delta // 60, "units")
        # )

        elements: list[ImageElentData] = []
        row = 0
        # facebox: np.ndarray[np._any, np.dtype[np.float32]]
        for path, facebox in zip(person.path, person.face_box):
            pilimg = PIL.Image.open(path)
            pil_img = PIL.ImageOps.exif_transpose(pilimg)
            img_width, img_height = pil_img.size

            (x1, y1, x2, y2) = facebox[0], facebox[1], facebox[2], facebox[3]

            box_w = x2 - x1
            box_h = y2 - y1
            pad_x = int(box_w * 0.2)
            pad_y = int(box_h * 0.2)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img_width, x2 + pad_x)
            y2 = min(img_height, y2 + pad_y)

            face_crop = pil_img.crop((x1, y1, x2, y2))

            img = resize_image(face_crop, 320, 200)
            tk_image_data = PIL.ImageTk.PhotoImage(img)

            col = 0
            image_label = ttk.Label(
                view, text=path, image=tk_image_data, compound="top"
            )
            image_label.grid(row=row, column=col, padx=10)

            image_tag_list = tk.Listbox(view, height=10)
            image_tag_list.grid(row=row, column=(col := col + 1), padx=10)
            row += 1  # noqa: SIM113

            current_tags = read_tags(path, self.et)
            image_tag_list.delete(0, "end")
            for tag in current_tags:
                image_tag_list.insert("end", tag)

            image_tag_list.bind(
                "<Button-1>",
                lambda e: self.select_listbox(e),
            )
            image_label.bind(
                "<Button-1>",
                lambda e: view_full_image_of_face(
                    e, self.cache_view_of_person_faces, self.index
                ),
            )

            _ = image_label.bind(
                "<MouseWheel>",
                lambda e: scroll.canvas.yview_scroll(-e.delta // 60, "units"),
            )

            elements.append(
                ImageElentData(tk_image_data, image_label, image_tag_list, False, path)
            )

        self.cache_view_of_person_faces.append(View(self.index, pview, elements))
        pview.tkraise()

    def select_listbox(self, e: tk.Event[tk.Listbox]):
        if self.curent_selected_listbox is not None:
            self.curent_selected_listbox.config(background=self.defaultbg)
        self.curent_selected_listbox = e.widget
        self.curent_selected_listbox.config(background="#FF9797")

    def parce_npz_file(
        self,
        npz_path: str | pathlib.Path,
        eps: float = 0.4,
        min_samples: int = 1,
    ):
        file_path = pathlib.Path(npz_path)
        data: np.lib.npyio.NpzFile = np.load(file_path, allow_pickle=True)

        image_paths: np.ndarray = data["paths"]
        bboxes: np.ndarray = data["bboxes"]

        if "embeddings" in data:
            embeddings: np.ndarray = data["embeddings"]

            if len(embeddings) == 0:
                raise Exception("No face embeddings found in the database.")  # noqa: TRY002

            clt = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
            clt.fit(embeddings)

            labels: np.ndarray = clt.labels_
            unique_labels: list[int] = sorted(set(labels))

            for label in unique_labels:
                if label == -1:
                    continue

                indices: np.ndarray = np.where(labels == label)[0]
                print(f"👤 Person #{label} ({len(indices)} images):")

                person = PersonFaces([], [])
                for idx in indices:
                    path: str = str(image_paths[idx])
                    box: np.ndarray = bboxes[idx]
                    person.path.append(path)
                    person.face_box.append(box)
                self.persons.append(person)
        else:
            for idx in range(len(image_paths)):
                path: str = str(image_paths[idx])
                box: np.ndarray = bboxes[idx]
                self.persons.append(PersonFaces([path], [box]))



def view_full_image_of_face(e: tk.Event[ttk.Label], views: list[View], index: int):
    for ele in views[index].elements:
        if ele.label == e.widget:
            os.startfile(ele.label.cget("text"))


def main():
    root = tk.Tk()

    selectec_npz_file = tkinter.filedialog.askopenfile(title="Pick a npz file")
    if selectec_npz_file is None or not selectec_npz_file.name.endswith(".npz"):
        tkinter.messagebox.showerror(
            "Pick a valid npz file",
            "Run `python main.py <folder>`, it will generate a valid file",
        )
        return

    with exiftool.ExifToolHelper() as et:
        app = TagEditorApp(root, et, selectec_npz_file.name)
        app.root.mainloop()


if __name__ == "__main__":
    main()
