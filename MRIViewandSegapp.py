import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import nibabel as nib
import os
import cv2
import asyncio
import threading


class MRIViewer:
    def __init__(self, root, event):
        self.root = root
        self.root.title("NIfTI MRI ve Çoklu Segmentasyon Görüntüleyici")

        # Set the theme and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Configure window
        self.root.geometry("1200x800")

        self.TARGET_SHAPE = (128, 128, 64)
        self.VOLUME_SHAPE = (512, 512, 64)
        self.DISPLAY_SIZE = (512, 512)

        self.mri_volume = None
        self.seg_volume_1 = None
        self.seg_volume_2 = None
        self.current_slice = 0

        self.WINDOW = 1000
        self.LEVEL = 0
        self.OPACITY = 0.5
        self.zoom_level = 1.0

        self.root.drop_target_register(DND_FILES)

        self.create_widgets()
        self.work_event = threading.Event()
        self.worker_thread = threading.Thread(target=self.worker, daemon=True)
        self.worker_thread.start()
        MRI_path=''

    def create_widgets(self):
        # Main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Control Frame
        control_frame = ctk.CTkFrame(self.main_container)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Button styles
        button_width = 160
        button_height = 35

        # Buttons with consistent styling
        ctk.CTkButton(control_frame,
                      text="MRI Yükle",
                      width=button_width,
                      height=button_height,
                      command=self.load_mri).pack(side=tk.LEFT, padx=5)

        ctk.CTkButton(control_frame,
                      text="Segmentasyon Yükle",
                      width=button_width,
                      height=button_height,
                      command=lambda: self.load_segmentation(1, '')).pack(side=tk.LEFT, padx=5)

        ctk.CTkButton(control_frame,
                      text="Segmentasyon Hesapla",
                      width=button_width,
                      height=button_height,
                      command=self.calculate_segmentation).pack(side=tk.LEFT, padx=5)

        ctk.CTkButton(control_frame,
                      text="Tümünü Temizle",
                      width=button_width,
                      height=button_height,
                      fg_color="darkred",
                      hover_color="red",
                      command=self.clear_all).pack(side=tk.LEFT, padx=5)

        # Zoom Frame
        zoom_frame = ctk.CTkFrame(self.main_container)
        zoom_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        ctk.CTkLabel(zoom_frame, text="Zoom:", font=("Helvetica", 12)).pack(side=tk.LEFT, padx=5)
        self.zoom_var = ctk.DoubleVar(value=1.0)
        self.zoom_slider = ctk.CTkSlider(zoom_frame,
                                         from_=0.5,
                                         to=4.0,
                                         variable=self.zoom_var,
                                         command=self.update_view)
        self.zoom_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # Slice Frame
        slice_frame = ctk.CTkFrame(self.main_container)
        slice_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ctk.CTkLabel(slice_frame, text="Kesit:", font=("Helvetica", 12)).pack(side=tk.LEFT, padx=5)
        self.slice_var = ctk.IntVar()
        self.slice_slider = ctk.CTkSlider(slice_frame,
                                          from_=0,
                                          to=self.TARGET_SHAPE[2] - 1,
                                          variable=self.slice_var,
                                          command=self.update_view)
        self.slice_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # Image Frame with modern styling
        self.image_frame = ctk.CTkFrame(self.main_container)
        self.image_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Image labels with backgrounds
        self.seg_label_1 = ctk.CTkLabel(self.image_frame, text="")
        self.seg_label_1.pack(side=tk.LEFT, padx=10, expand=True)

        self.mri_label = ctk.CTkLabel(self.image_frame, text="")
        self.mri_label.pack(side=tk.LEFT, padx=10, expand=True)

        self.seg_label_2 = ctk.CTkLabel(self.image_frame, text="")
        self.seg_label_2.pack(side=tk.LEFT, padx=10, expand=True)

    def calculate_segmentation(self):
        self.work_event.set()

    def load_segmentation(self, seg_num,file_path):
        if file_path == '':
            file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii *.nii.gz")])
        if file_path:
            try:
                nifti = nib.load(file_path)
                volume = nifti.get_fdata()
                volume = self.resize_volume(volume)

                if seg_num == 1:
                    self.seg_volume_1 = volume
                else:
                    self.seg_volume_2 = volume

                self.update_view()
                if seg_num == 1:
                    messagebox.showinfo("Başarılı", f"Segmentasyon başarıyla yüklendi")
                else:
                    messagebox.showinfo("Başarılı", f"Segmentasyon başarıyla hesaplandı")
            except Exception as e:
                messagebox.showerror("Hata", f"Segmentasyon yüklenemedi: {str(e)}")

    def update_view(self, *args):
        if self.mri_volume is None:
            return

        current_slice = self.slice_var.get()
        zoom = self.zoom_var.get()
        new_size = (int(self.DISPLAY_SIZE[0] * zoom), int(self.DISPLAY_SIZE[1] * zoom))

        # MRI kesiti

        mri_slice = self.mri_volume[:, :, current_slice].copy()
        mri_display = self.prepare_slice_for_display(mri_slice)
        mri_image = Image.fromarray(mri_display)
        mri_image = mri_image.resize(new_size, Image.LANCZOS)
        mri_photo = ImageTk.PhotoImage(mri_image)
        self.mri_label.configure(image=mri_photo)
        self.mri_label.image = mri_photo

        # Segmentasyon 1
        if self.seg_volume_1 is not None:
            seg_slice = self.seg_volume_1[:, :, current_slice].copy()
            seg_display = self.prepare_slice_for_display(seg_slice, is_seg=True)
            blended = cv2.addWeighted(mri_display, 1 - self.OPACITY, seg_display, self.OPACITY, 0)
            seg_image = Image.fromarray(blended)
            seg_image = seg_image.resize(new_size, Image.LANCZOS)
            seg_photo = ImageTk.PhotoImage(seg_image)
            self.seg_label_1.configure(image=seg_photo)
            self.seg_label_1.image = seg_photo

        # Segmentasyon 2
        if self.seg_volume_2 is not None:
            seg_slice = self.seg_volume_2[:, :, current_slice].copy()
            seg_display = self.prepare_slice_for_display(seg_slice, is_seg=True)
            blended = cv2.addWeighted(mri_display, 1 - self.OPACITY, seg_display, self.OPACITY, 0)
            seg_image = Image.fromarray(blended)
            seg_image = seg_image.resize(new_size, Image.LANCZOS)
            seg_photo = ImageTk.PhotoImage(seg_image)
            self.seg_label_2.configure(image=seg_photo)
            self.seg_label_2.image = seg_photo

    def clear_all(self):
        self.mri_volume = None
        self.seg_volume_1 = None
        self.seg_volume_2 = None
        self.current_slice = 0
        self.slice_var.set(0)
        self.zoom_var.set(1.0)
        self.mri_label.configure(image='')
        self.seg_label_1.configure(image='')
        self.seg_label_2.configure(image='')
        messagebox.showinfo("Bilgi", "Tüm görüntüler temizlendi")

    # Diğer metodlar aynı kalacak (resize_volume, prepare_slice_for_display, apply_window_level, vb.)
    def resize_volume(self, volume):

        self.TARGET_SHAPE = (512, 512, 64)


        resized = np.zeros(self.TARGET_SHAPE)
        for i in range(volume.shape[2]):
            if i >= self.TARGET_SHAPE[2]:
                break

            slice_2d = volume[:, :, i]
            resized_slice = cv2.resize(slice_2d,
                                       (self.TARGET_SHAPE[0], self.TARGET_SHAPE[1]),
                                       interpolation=cv2.INTER_LINEAR)
            resized[:, :, i] = resized_slice

        if volume.shape[2] != self.TARGET_SHAPE[2]:
            z_scale = self.TARGET_SHAPE[2] / volume.shape[2]
            indices = np.arange(self.TARGET_SHAPE[2]) / z_scale
            indices = np.minimum(indices.astype(np.int32), volume.shape[2] - 1)
            resized = resized[:, :, indices]

        return resized

    def prepare_slice_for_display(self, slice_data, is_seg=False):
        if not is_seg:
            slice_data = self.apply_window_level(slice_data)
            display = (slice_data * 255).astype(np.uint8)
            display = np.stack([display] * 3, axis=-1)
        else:
            display = np.zeros((slice_data.shape[0], slice_data.shape[1], 3), dtype=np.uint8)
            unique_labels = np.unique(slice_data)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

            for i, label in enumerate(unique_labels[1:]):
                color = colors[i % len(colors)]
                display[slice_data == label] = color

        return display

    def apply_window_level(self, image):
        min_value = self.LEVEL - self.WINDOW / 2
        max_value = self.LEVEL + self.WINDOW / 2
        return np.clip((image - min_value) / (max_value - min_value), 0, 1)

    def load_mri(self):
        MRI_path = filedialog.askopenfilename(
            filetypes=[("NIfTI files", "*.nii *.nii.gz")])
        self.MRI_path=MRI_path
        if MRI_path:
            try:
                nifti = nib.load(MRI_path)
                volume = nifti.get_fdata()
                #self.mri_volume = volume
                self.mri_volume = self.resize_volume(volume)
                self.update_view()
                messagebox.showinfo("Başarılı", "MRI başarıyla yüklendi")
            except Exception as e:
                messagebox.showerror("Hata", f"MRI yüklenemedi: {str(e)}")

    def worker(self):
        print("Worker started")
        from monai.utils import first, set_determinism
        from monai.transforms import (
            Compose,
            AddChanneld,
            LoadImaged,
            Resized,
            ToTensord,
            Spacingd,
            Orientationd,
            ScaleIntensityRanged,
            CropForegroundd,
            Activations,
        )

        from monai.networks.nets import UNet
        from monai.networks.layers import Norm
        from monai.data import CacheDataset, DataLoader, Dataset

        import torch

        import numpy as np

        from monai.inferers import sliding_window_inference
        import nibabel as nib
        import torchvision.transforms as T

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU is not available")

        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
        model.load_state_dict(torch.load("C:/Users/HP/Desktop/kaan/Ödevler/Bitirme/TrSonuclar/best_metric_model.pth"))
        model.eval()

        sw_batch_size = 4
        roi_size = (128, 128, 64)
        print("Model loaded")
        while True:
            self.work_event.wait()
            print("Event set")
            with torch.no_grad():
                file_path = self.MRI_path
                set_determinism(seed=0)
                files = [{"vol": file_path}]
                pixdim = (1.5, 1.5, 1.0)
                a_min = -200
                a_max = 200
                spatial_size = [128, 128, 64]
                cache = True

                transforms = Compose([
                    LoadImaged(keys=["vol"]),
                    AddChanneld(keys=["vol"]),
                    Spacingd(keys=["vol"], pixdim=pixdim, mode="bilinear"),
                    Orientationd(keys=["vol"], axcodes="RAS"),
                    ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["vol"], source_key="vol"),
                    Resized(keys=["vol"], spatial_size=spatial_size),
                    ToTensord(keys=["vol"]),
                ])

                if cache:
                    ds = CacheDataset(data=files, transform=transforms, cache_rate=1.0)
                else:
                    ds = Dataset(data=files, transform=transforms)

                t_volume = DataLoader(ds, batch_size=1)
                t_volume = first(t_volume)

                test_outputs = sliding_window_inference(t_volume['vol'].to(device), roi_size, sw_batch_size, model)
                sigmoid_activation = Activations(sigmoid=True)
                test_outputs = sigmoid_activation(test_outputs)
                test_outputs = test_outputs > 0.53

            new_channel = torch.zeros(128, 128, 1)

            test_outputs = test_outputs.to(torch.device("cpu"))
            denenen = torch.cat((new_channel, test_outputs[0, 1, :, :, :]), dim=2)

            kullanılan = denenen.permute(2, 0, 1)

            # Y ekseninde sıkıştırma
            new_height = 128
            new_width = 64
            resized_image = T.Resize((new_height, new_width))(kullanılan)

            # Yeni bir boş tuval oluştur (orijinal boyut, sıfırlarla doldurulmuş)
            padded_image = torch.zeros_like(kullanılan)

            # Sıkıştırılmış görseli merkeze yerleştir
            start_y = (padded_image.shape[1] - resized_image.shape[1]) // 2
            start_x = (padded_image.shape[2] - resized_image.shape[2]) // 2

            padded_image[:, start_y:start_y + resized_image.shape[1], start_x:start_x + resized_image.shape[2]] = resized_image
            padded_image = padded_image.permute(1, 2, 0)
            padded_image = np.where(padded_image > 0, 1, 0)
            nifti_image = nib.Nifti1Image(padded_image[::-1, :, :], affine=np.eye(4))
            seg_volume_2 = nifti_image


            nib.save(nifti_image, 'C:/Users/HP/Desktop/seg_tahmini_005_0.nii')
            file_path='C:/Users/HP/Desktop/seg_tahmini_005_0.nii'
            self.load_segmentation(2,file_path)
            self.work_event.clear()


def main():

    global event, root, task
    event = asyncio.Event()
    root = TkinterDnD.Tk()
    app = MRIViewer(root,event)
    print("Main started")
    root.mainloop()


if __name__ == "__main__":
    main()