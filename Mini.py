import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import *
from tkinter import messagebox
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, lit, when, col
import matplotlib.pyplot as plt
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from PIL import Image, ImageTk
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import ClusteringEvaluator, MulticlassClassificationEvaluator
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Assuming you have a SparkSession already created
spark = SparkSession.builder.appName("example").getOrCreate()

window = tk.Tk()

window.title("ML Models Analysis")
wrapper1 = LabelFrame(window, text="Complete Data")
wrapper1.pack(fill="both", expand="no", padx=20, pady=10)

wrapper2 = LabelFrame(window, text="Network Analysis")
wrapper2.pack(fill="both", expand="no", padx=20, pady=10)

wrapper3 = LabelFrame(window, text="Data Analysis")
wrapper3.pack(fill="both", expand="no", padx=20, pady=10)

wrapper4 = LabelFrame(window, text="Plotting Data")
wrapper4.pack(fill="both", expand="no", padx=20, pady=10)


def load_data():
    XSS_df = spark.read.csv('./dataset/Dataset_WAF/XSS_dataset.csv', header=True, escape="\"")
    Http_Params_df = spark.read.csv('./dataset/Dataset_WAF/payload_full.csv', header=True, escape="\"")
    Emcl_data = spark.read.json('./dataset/Dataset_WAF/ecml_data.json')

    XSS_df_with_type = (
        XSS_df.withColumn("Sentence", concat(col("Sentence"), lit("xss")))
        .withColumn("Label", when(col("Label") == 1, "xss").otherwise("valid"))
        .drop("_c0")
    )

    Http_Params_df_processed = (
        Http_Params_df
        .withColumn("attack_type", when(col("attack_type") == "norm", "valid")
                    .otherwise(col("attack_type")))
        .drop("length")
        .drop("label")
    )

    complete_df = (
        XSS_df_with_type.select("Sentence", "Label")
        .union(Emcl_data.select("pattern", "type").withColumnRenamed("pattern", "Sentence")
               .withColumnRenamed("type", "Label"))
        .union(Http_Params_df_processed.select("payload", "attack_type")
               .withColumnRenamed("pattern", "Sentence").withColumnRenamed("type", "Label"))
    )

    return complete_df


def ButtonClicker():
    complete_df = load_data()
    counts = complete_df.groupBy("Label").count().orderBy("count", ascending=False)
    label_counts = counts.collect()
    labels = [row["Label"] for row in label_counts]
    counts = [row["count"] for row in label_counts]
    plt.pie(counts, labels=labels, startangle=90)
    plt.title("Label Distribution")
    plt.savefig('Distributions.png')
    render_image('Distributions.png')
    plt.show()


def render_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((200, 200))
    photo = ImageTk.PhotoImage(image)
    img_label = tk.Label(wrapper4, image=photo)
    img_label.image = photo
    img_label.grid(row=0, column=0, columnspan=2)


def Kmeanss():
    messagebox.showinfo('information', 'Please be patient this might take a while')
    loop.run_in_executor(executor, kmeans_task)


def kmeans_task():
    transformed_data = transform_Data()
    kmeans = KMeans(k=5, featuresCol="features", predictionCol="prediction")
    model = kmeans.fit(transformed_data.get('transformed_data'))
    clustered_data = model.transform(transformed_data.get('transformed_data'))
    evaluator = ClusteringEvaluator()
    score = evaluator.evaluate(clustered_data) * 100
    l = Label(wrapper3, text="Kmeans Accuracy=")
    l.grid(row=2, column=1)
    acc = Label(wrapper3, text=score)
    acc.grid(row=2, column=2)


def LogisticRegressions():
    messagebox.showinfo('information', 'Please be patient this might take a while')
    loop.run_in_executor(executor, logistic_task)


def logistic_task():
    transformed_data = transform_Data()
    lr = LogisticRegression(
        featuresCol="features", labelCol="Label", maxIter=10)
    lr_model = lr.fit(transformed_data.get('train_data'))
    lr_predictions = lr_model.transform(transformed_data.get('test_data'))
    lr_evaluator = MulticlassClassificationEvaluator(
        labelCol="Label", predictionCol="prediction", metricName="accuracy")
    score = lr_evaluator.evaluate(lr_predictions) * 100
    l = Label(wrapper3, text="Logistic Regression Accuracy = ")
    l.grid(row=3, column=1)
    acc = Label(wrapper3, text=score)
    acc.grid(row=3, column=2)


def transform_Data():
    complete_df = load_data()
    complete_df_numeric_labels = complete_df.withColumn(
        "Label", when(col("Label") == "valid", 0)
        .when(col("Label") == "sqli", 1)
        .when(col("Label") == "xss", 2)
        .when(col("Label") == "cmdi", 3)
        .when(col("Label") == "path-traversal", 4)
        .otherwise(col("Label").cast("int"))
    )
    tokenizer = Tokenizer(inputCol="Sentence", outputCol="words")
    vectorizer = CountVectorizer(inputCol="words", outputCol="features")
    pipes = Pipeline(stages=[tokenizer, vectorizer])
    transformed_Data = pipes.fit(
        complete_df_numeric_labels).transform(complete_df_numeric_labels)
    (train_data, test_data) = transformed_Data.randomSplit([0.8, 0.2], seed=123)

    return {'transformed_data': transformed_Data, 'train_data': train_data, 'test_data': test_data}


def The_search_filter():
    complete_Data = load_data()
    inputtxt = search_filter.get("1.0", "end-1c").strip().lower()

    if inputtxt == 'xss':
        result = complete_Data.filter(col('Label') == 'xss').limit(20)

    elif inputtxt == 'valid':
        result = complete_Data.filter(col('Label') == 'valid').limit(20)
    elif inputtxt == 'sqli' or inputtxt == 'sql injection':
        result = complete_Data.filter(col('Label') == 'sqli').limit(20)
    elif inputtxt == 'path':
        result = complete_Data.filter(col('Label') == 'path-travesal').limit(20)
    elif inputtxt == 'cmd':
        result = complete_Data.filter(col('Label') == 'cmdi').limit(20)
    else:
        messagebox.showinfo('Search Result', 'Query not found')
        return

    # Display the result in a Treeview in wrapper3
    tree_result = ttk.Treeview(wrapper3, columns=("Sentence", "Label"), show="headings")
    tree_result.column("Sentence", anchor=tk.W, width=300)
    tree_result.column("Label", anchor=tk.W, width=100)
    tree_result.heading("Sentence", text="Sentence")
    tree_result.heading("Label", text="Label")

    # Insert data into the Treeview
    for row in result.collect():
        tree_result.insert("", tk.END, values=row)

    tree_result.grid(row=7, column=3, sticky="nsew")
    wrapper3.columnconfigure(3, weight=1)
    wrapper3.rowconfigure(7, weight=1)


def total_count():
    complete_Data = load_data()
    result = complete_Data.groupBy('label').count().orderBy("count", ascending=True)
    messagebox.showinfo("Total Count", result.collect())


# Replace the following line with your actual data loading logic
Complete_Data = load_data()

# Create a Treeview widget
tree = ttk.Treeview(wrapper1, columns=(1, 2), show="headings", height=6)

tree["columns"] = ("Sentence", "Label")

# Define columns
tree.column("#0", width=0, stretch=tk.NO)
tree.column("Sentence", anchor=tk.W, width=300)
tree.column("Label", anchor=tk.W, width=100)

# Define column headings
tree.heading("#0", text="", anchor=tk.W)
tree.heading("Sentence", text="Sentence")
tree.heading("Label", text="Label")

# Insert data into the Treeview
for row in Complete_Data.collect():
    tree.insert("", tk.END, values=row)

tree.pack(padx=40, pady=5, fill=tk.BOTH, expand=True)

search_filter = Text(wrapper3, height=1, width=25)
search_filter.grid(row=0, column=1, padx=5)
search_btn = Button(wrapper3, text="Search", command=The_search_filter)
search_btn.grid(row=0, column=2, padx=5)
plot_button = tk.Button(
    wrapper2,
    text='Plot Labels',
    command=ButtonClicker
)
plot_button.pack(side=tk.LEFT, padx=10, pady=10)
lr_button = tk.Button(
    wrapper2,
    text='Logistic Regressions',
    command=LogisticRegressions
)
lr_button.pack(side=tk.LEFT, padx=10, pady=10)
kmeans_button = tk.Button(
    wrapper2,
    text='Kmeans',
    command=Kmeanss
)
kmeans_button.pack(side=tk.LEFT, padx=10, pady=10)
total_counts = tk.Button(
    wrapper2,
    text='Total Count',
    command=total_count
)
total_counts.pack(side=tk.LEFT, padx=10, pady=10)

window.geometry("800x800")

# Create an executor for running blocking tasks in a separate thread
executor = ThreadPoolExecutor(max_workers=1)

# Create an event loop
loop = asyncio.get_event_loop()

# Start the Tkinter main event loop
window.mainloop()
