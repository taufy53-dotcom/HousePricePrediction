import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
from sklearn.linear_model import Ridge
import os
import sys

def resource_path(relative_path):
    """Get the correct path for normal Python and PyInstaller EXE."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



# ============================================================
# CONFIGURATION
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    resource_path("data/house_dataset.csv")
)


# ============================================================
# PREPROCESSING
# ============================================================

binary_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

for col in binary_columns:
    df[col] = df[col].map({
        "yes": 1,
        "no": 0
    })


df = pd.get_dummies(
    df,
    columns=["furnishingstatus"],
    drop_first=True
)


X = df.drop("price", axis=1)
y = df["price"]


# ============================================================
# TRAIN MODEL
# ============================================================

model = Ridge(alpha=0.01)
model.fit(X, y)


# ============================================================
# GUI
# ============================================================

class HousePriceApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("House Price Predictor")
        self.geometry("1100x720")
        self.minsize(900, 600)

        self.bg_color = "#0B1120"
        self.card_color = "#111827"
        self.input_color = "#1F2937"
        self.accent_color = "#3B82F6"
        self.text_color = "#F8FAFC"
        self.secondary_text = "#94A3B8"

        self.configure(fg_color=self.bg_color)

        # ====================================================
        # MAIN WINDOW
        # ====================================================

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)


        # ====================================================
        # LEFT SCROLLABLE AREA
        # ====================================================

        self.left_scroll = ctk.CTkScrollableFrame(
            self,
            fg_color=self.bg_color,
            scrollbar_button_color="#374151",
            scrollbar_button_hover_color="#4B5563"
        )

        self.left_scroll.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(30, 10),
            pady=25
        )

        self.left_scroll.grid_columnconfigure(0, weight=1)
        self.left_scroll.grid_columnconfigure(1, weight=1)


        # ====================================================
        # TITLE
        # ====================================================

        title = ctk.CTkLabel(
            self.left_scroll,
            text="🏠  House Price Predictor",
            font=ctk.CTkFont(
                size=30,
                weight="bold"
            ),
            text_color=self.text_color
        )

        title.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            padx=10,
            pady=(5, 5)
        )


        subtitle = ctk.CTkLabel(
            self.left_scroll,
            text="Enter property details to estimate its market price",
            font=ctk.CTkFont(size=14),
            text_color=self.secondary_text
        )

        subtitle.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            padx=10,
            pady=(0, 25)
        )


        # ====================================================
        # INPUTS
        # ====================================================

        self.area_entry = self.create_input(
            "Area (sq ft)",
            2,
            0,
            "e.g. 7420"
        )

        self.bedrooms_entry = self.create_input(
            "Bedrooms",
            2,
            1,
            "e.g. 4"
        )

        self.bathrooms_entry = self.create_input(
            "Bathrooms",
            4,
            0,
            "e.g. 2"
        )

        self.stories_entry = self.create_input(
            "Stories",
            4,
            1,
            "e.g. 3"
        )

        self.parking_entry = self.create_input(
            "Parking Spaces",
            6,
            0,
            "e.g. 2"
        )

        self.mainroad_menu = self.create_dropdown(
            "Main Road",
            6,
            1
        )

        self.guestroom_menu = self.create_dropdown(
            "Guest Room",
            8,
            0
        )

        self.basement_menu = self.create_dropdown(
            "Basement",
            8,
            1
        )

        self.hotwater_menu = self.create_dropdown(
            "Hot Water Heating",
            10,
            0
        )

        self.airconditioning_menu = self.create_dropdown(
            "Air Conditioning",
            10,
            1
        )

        self.prefarea_menu = self.create_dropdown(
            "Preferred Area",
            12,
            0
        )


        # ====================================================
        # FURNISHING
        # ====================================================

        self.create_label(
            "Furnishing Status",
            12,
            1
        )

        self.furnishing_menu = ctk.CTkComboBox(
            self.left_scroll,
            values=[
                "furnished",
                "semi-furnished",
                "unfurnished"
            ],
            height=42,
            corner_radius=8,
            fg_color=self.input_color,
            border_color="#374151",
            button_color=self.accent_color,
            text_color=self.text_color,
            dropdown_fg_color=self.card_color
        )

        self.furnishing_menu.set("unfurnished")

        self.furnishing_menu.grid(
            row=13,
            column=1,
            sticky="ew",
            padx=10,
            pady=(0, 20)
        )


        # ====================================================
        # PREDICT BUTTON
        # ====================================================

        self.predict_button = ctk.CTkButton(
            self.left_scroll,
            text="💰  PREDICT HOUSE PRICE",
            height=52,
            corner_radius=10,
            font=ctk.CTkFont(
                size=16,
                weight="bold"
            ),
            fg_color=self.accent_color,
            hover_color="#2563EB",
            command=self.predict_price
        )

        self.predict_button.grid(
            row=14,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=10,
            pady=(10, 25)
        )


        # ====================================================
        # RIGHT RESULT CARD
        # ====================================================

        self.right_frame = ctk.CTkFrame(
            self,
            fg_color=self.card_color,
            corner_radius=20
        )

        self.right_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(10, 30),
            pady=25
        )


        # ====================================================
        # PREDICTION HEADER
        # ====================================================

        self.result_heading = ctk.CTkLabel(
            self.right_frame,
            text="PREDICTION",
            font=ctk.CTkFont(
                size=14,
                weight="bold"
            ),
            text_color=self.accent_color
        )

        self.result_heading.pack(
            pady=(70, 10)
        )


        # ====================================================
        # HOUSE ICON
        # ====================================================

        self.house_icon = ctk.CTkLabel(
            self.right_frame,
            text="🏡",
            font=ctk.CTkFont(size=65)
        )

        self.house_icon.pack(
            pady=(5, 15)
        )


        # ====================================================
        # RESULT TITLE
        # ====================================================

        self.result_title = ctk.CTkLabel(
            self.right_frame,
            text="Estimated House Price",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            text_color=self.text_color
        )

        self.result_title.pack()


        # ====================================================
        # PRICE
        # ====================================================

        self.price_label = ctk.CTkLabel(
            self.right_frame,
            text="₹ --",
            font=ctk.CTkFont(
                size=38,
                weight="bold"
            ),
            text_color=self.text_color
        )

        self.price_label.pack(
            pady=(20, 10)
        )


        # ====================================================
        # STATUS
        # ====================================================

        self.status_label = ctk.CTkLabel(
            self.right_frame,
            text="Enter property details\nand click Predict",
            font=ctk.CTkFont(size=14),
            text_color=self.secondary_text,
            justify="center"
        )

        self.status_label.pack(
            pady=10
        )


        # ====================================================
        # MODEL CARD
        # ====================================================

        self.model_card = ctk.CTkFrame(
            self.right_frame,
            fg_color=self.input_color,
            corner_radius=12
        )

        self.model_card.pack(
            fill="x",
            padx=30,
            pady=(45, 0)
        )


        model_label = ctk.CTkLabel(
            self.model_card,
            text="MODEL INFORMATION",
            font=ctk.CTkFont(
                size=12,
                weight="bold"
            ),
            text_color=self.secondary_text
        )

        model_label.pack(
            pady=(15, 5)
        )


        model_name = ctk.CTkLabel(
            self.model_card,
            text="Ridge Regression",
            font=ctk.CTkFont(
                size=16,
                weight="bold"
            ),
            text_color=self.text_color
        )

        model_name.pack(
            pady=(0, 5)
        )


        alpha_label = ctk.CTkLabel(
            self.model_card,
            text="Alpha: 0.01",
            font=ctk.CTkFont(size=13),
            text_color=self.secondary_text
        )

        alpha_label.pack(
            pady=(0, 15)
        )


    # ========================================================
    # LABEL
    # ========================================================

    def create_label(self, text, row, column):

        label = ctk.CTkLabel(
            self.left_scroll,
            text=text,
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            ),
            text_color=self.text_color
        )

        label.grid(
            row=row,
            column=column,
            sticky="w",
            padx=10,
            pady=(0, 5)
        )

        return label


    # ========================================================
    # INPUT
    # ========================================================

    def create_input(
        self,
        label_text,
        row,
        column,
        placeholder
    ):

        self.create_label(
            label_text,
            row,
            column
        )

        entry = ctk.CTkEntry(
            self.left_scroll,
            height=42,
            corner_radius=8,
            fg_color=self.input_color,
            border_color="#374151",
            text_color=self.text_color,
            placeholder_text=placeholder,
            placeholder_text_color="#64748B"
        )

        entry.grid(
            row=row + 1,
            column=column,
            sticky="ew",
            padx=10,
            pady=(0, 15)
        )

        return entry


    # ========================================================
    # DROPDOWN
    # ========================================================

    def create_dropdown(
        self,
        label_text,
        row,
        column
    ):

        self.create_label(
            label_text,
            row,
            column
        )

        menu = ctk.CTkComboBox(
            self.left_scroll,
            values=["yes", "no"],
            height=42,
            corner_radius=8,
            fg_color=self.input_color,
            border_color="#374151",
            button_color=self.accent_color,
            text_color=self.text_color,
            dropdown_fg_color=self.card_color
        )

        menu.set("no")

        menu.grid(
            row=row + 1,
            column=column,
            sticky="ew",
            padx=10,
            pady=(0, 15)
        )

        return menu


    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_price(self):

        try:

            # ------------------------------------------------
            # NUMERICAL INPUTS
            # ------------------------------------------------

            area = float(
                self.area_entry.get()
            )

            bedrooms = int(
                self.bedrooms_entry.get()
            )

            bathrooms = int(
                self.bathrooms_entry.get()
            )

            stories = int(
                self.stories_entry.get()
            )

            parking = int(
                self.parking_entry.get()
            )


            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if area <= 0:
                raise ValueError(
                    "Area must be greater than 0."
                )

            if bedrooms <= 0:
                raise ValueError(
                    "Bedrooms must be greater than 0."
                )

            if bathrooms <= 0:
                raise ValueError(
                    "Bathrooms must be greater than 0."
                )

            if stories <= 0:
                raise ValueError(
                    "Stories must be greater than 0."
                )

            if parking < 0:
                raise ValueError(
                    "Parking spaces cannot be negative."
                )


            # ------------------------------------------------
            # CATEGORICAL INPUTS
            # ------------------------------------------------

            mainroad = self.mainroad_menu.get()
            guestroom = self.guestroom_menu.get()
            basement = self.basement_menu.get()
            hotwaterheating = self.hotwater_menu.get()
            airconditioning = self.airconditioning_menu.get()
            prefarea = self.prefarea_menu.get()

            furnishingstatus = (
                self.furnishing_menu.get()
            )


            # ------------------------------------------------
            # CREATE DATAFRAME
            # ------------------------------------------------

            input_data = pd.DataFrame({

                "area": [area],

                "bedrooms": [bedrooms],

                "bathrooms": [bathrooms],

                "stories": [stories],

                "mainroad": [mainroad],

                "guestroom": [guestroom],

                "basement": [basement],

                "hotwaterheating": [
                    hotwaterheating
                ],

                "airconditioning": [
                    airconditioning
                ],

                "parking": [parking],

                "prefarea": [prefarea],

                "furnishingstatus": [
                    furnishingstatus
                ]

            })


            # ------------------------------------------------
            # YES / NO → 1 / 0
            # ------------------------------------------------

            for col in binary_columns:

                input_data[col] = (
                    input_data[col]
                    .map({
                        "yes": 1,
                        "no": 0
                    })
                )


            # ------------------------------------------------
            # ONE-HOT ENCODING
            # ------------------------------------------------

            input_data = pd.get_dummies(
                input_data,
                columns=["furnishingstatus"],
                drop_first=True
            )


            # ------------------------------------------------
            # MATCH TRAINING COLUMNS
            # ------------------------------------------------

            input_data = input_data.reindex(
                columns=X.columns,
                fill_value=0
            )


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            predicted_price = model.predict(
                input_data
            )[0]


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            self.price_label.configure(
                text=f"₹ {predicted_price:,.2f}"
            )

            self.status_label.configure(
                text="Prediction generated successfully!",
                text_color="#22C55E"
            )


        except ValueError as e:

            messagebox.showerror(
                "Invalid Input",
                str(e)
            )


        except Exception as e:

            messagebox.showerror(
                "Prediction Error",
                f"Something went wrong:\n\n{e}"
            )


# ============================================================
# START APP
# ============================================================

if __name__ == "__main__":

    app = HousePriceApp()

    app.mainloop()
