import pandas as pd

from geosc.plotting import NaiveBayesGaussianContour2D


# ===== input data =====
csv_path = "/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/WellLog/WellLogCNOOC_CNOOC-2D.csv"
feature_cols = ["AI", "SI"]
target_col = "LithoId"
skiprows = [1]

# class yang dipakai di training/plot
use_classes = [1, 2, 3, 4, 5]

# nama class dari user
# format: Litho_Id -> Litho_Name
class_names = {
    1: "Shale",
    2: "SandStone",
    3: "Coal",
    4: "LimeStone",
    5: "Basement",
}

# warna class (opsional)
class_colors = {
    1: "green",
    2: "yellow",
    3: "black",
    4: "blue",
    5: "red",
}


df = pd.read_csv(csv_path, header=0, skiprows=skiprows)

plotter = NaiveBayesGaussianContour2D(
    class_names=class_names,
    class_colors=class_colors,
    null_value=-999.25,
    var_smoothing=1e-9,
)

plotter.fit_from_dataframe(
    df=df,
    feature_cols=feature_cols,
    target_col=target_col,
    allowed_classes=use_classes,
)

# plotter.plot(title="Gaussian Contours Naive Bayes (2D)")
plotter.plot(
    title="Gaussian Contours Naive Bayes (2D)",
    grid_size=300,
    contour_count=6,
    percentiles=(70.0, 99.5),
    show_points=True,
)