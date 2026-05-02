"""Model training and evaluation helpers."""

from __future__ import annotations
from dataclasses import dataclass
""" data class result modealkany tedaya"""
from typing import Any
""" bo typakan lanau modlakan bakar de"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
""" bo peuanay har yakanay manay chya wrdbynay accuracy har yakayn
katek modeelaka peuanay naxoshy daa chand jar rast
seyakm  chand naxoshy rastyqna dozraua

recal score chand halaty naxohsy reastaqyna dozrayaua
last one quality rezbandyakan dapeua barztr bahsta"""
from sklearn.naive_bayes import GaussianNB
""" au modelakany problity bakar dene fast balam simple"""
from sklearn.neighbors import KNeighborsClassifier
""" peshbyny daka ba psht bastn ba  nzyktryn wenay hausheua"""
from sklearn.svm import SVC
""" ba psht bastn bau vectoranay ba amer peuana krayan"""
from sklearn.preprocessing import label_binarize


@dataclass
class EvaluationResult:
    """ bo halgrtny anjamy halsangand """
    model_name: str
    accuracy: float
    precision: float
    """ wrdbyny modelaka nmuna 20 ruak naxoshy ahay au 16 garndauatau"""
    recall: float
    """ wabirhenanan lanau hamu naxoshakan ruakan 
    modleakan chandyan dozyaua"""
    f1: float
    """ balancy newuan wrdbynuy wabir henanauay"""
    roc_auc: float | None = None
    """ model chand bash polakan jiadakatau barztr bashtra"""

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "model": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }
    """ data dagory bo table dict[strin float none data type]
    return maby some thing like this
    {
 "model": "CNN",
 "accuracy": 0.97,
 "precision": 0.96,
 "recall": 0.97,
 "f1": 0.96,
 "roc_auc": 0.99
}"""
 
"""drusktnry rahanany model knn bakar henany data rahenrau  bakar henanu taybatmandy lely galay bameu """
def train_knn(x_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 7) -> KNeighborsClassifier:
    """ xtrain input taybat mandy  rahenrau la x-train row image column dakre rgb auana bet
    ytrain datay esh lasarkrau ja  healthy any one
    au 7 nzyktryn 7 jiran hsab da"""
    """ katek wenay nwe hat 7 nzytkryn wena lek dadataua"""
    """ drause nyzkakan grngy zyatr wardagrn"""
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    model.fit(x_train, y_train)
    """ nmunakan labir daka zyatr datay rahnerau haldagre"""
    return model
    """ modely rahenrau dagarenetaua
    nmuna agar 7 sample 6 helghty yakay naxosh aua helahy"""
    """ esta model pridic lo bakar henana"""



def train_gaussian_nb(x_train: np.ndarray, y_train: np.ndarray) -> GaussianNB:
    """ xtrian taybatamany   gala ytrain nauy naxoshy rast"""
    model = GaussianNB()
    """ ba bakar henany nmra pekhautau auana agar daka """
    """sckitl learn modely amadakrau pe wysta modelakan la sfr drus bky
    labiry auay xot birkayana au esha bkay modely hazr bakary de"""
    model.fit(x_train, y_train)
    return model


""" zor jar anjameky bahez dadda lasar tabat mandya dastyakan"""
def train_svm_rbf(x_train: np.ndarray, y_train: np.ndarray, c_value: float = 10.0) -> SVC:
    """ x hamysha featuerakay y  rahenrauakay"""
    """ floag 10 halajan control daka snury tund """
    model = SVC(kernel="rbf", C=c_value, probability=True, class_weight="balanced")
    """ rega ba snury bryardna line kehsrau dada rega ba snury bryardnay  keshraua 
    katak data kan saxu anay tr tekal bun rbf snur lanuanyan dakesh
    balance drust bka laneaun wenay konu xxor """
    model.fit(x_train, y_train)
    """babakar henany handcrafte """
    return model




""" au functina bakar de bo taqy krndauy modely rahenrau
duay train krnd modeklakan ta chand bashn bo data nabinrauakan
input xtest y test output pridciotn yan probley if have"""
""" this ffunciton evaluate which mode is best """



""" au function evaluate modekala daka ka kamay basthr"""
def evaluate_classifier(model: Any, x_test: np.ndarray, y_test: np.ndarray, class_count: int | None = None) -> tuple[EvaluationResult, np.ndarray, np.ndarray | None]:
    predictions = model.predict(x_test)
    """ daua la model daka bo peshbnyn krny modely taqy rkd"""
    probabilities = None
    roc_auc = None
    """ modelaka chand bash classkan jiadata"""

    if hasattr(model, "predict_proba"):
        """ aya modelaaka arky agry haya yaxud na hamuayn hayan svm  agar chalak krabe"""
        probabilities = model.predict_proba(x_test)
        """Get probabilities.

Example:

Healthy = 0.90
Mildew = 0.05
Anthracnose = 0.05"""
    elif hasattr(model, "decision_function"):
        """ aya modleaka nmary bryar dany haya yaxud na"""
        decision_values = model.decision_function(x_test)
        if decision_values.ndim == 1:
            """ anjamy brayary yak dimention daykaya du dimention"""
            probabilities = np.column_stack([1 - decision_values, decision_values])
        else:
            probabilities = decision_values

    average = "weighted"
    """ auany numnay zyatrayn haya keshy restuly ba dast bena basha bo data na hausangakan"""
    result = EvaluationResult(
        model_name=model.__class__.__name__,
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, average=average, zero_division=0)),
        recall=float(recall_score(y_test, predictions, average=average, zero_division=0)),
        
        f1=float(f1_score(y_test, predictions, average=average, zero_division=0)),

    )
    """ zero division wata dur bkaua la hala agar preidiont sfr habe"""


    """ bashy modealkan lanau polakan jia dakatau"""
    if probabilities is not None and class_count is not None:
        try:
            classes = np.arange(class_count)
            """ auayn bo drust krndy class numberka"""
            y_test_binarized = label_binarize(y_test, classes=classes)
            """ au labelanay ka ahaya ayankata binery
             nmuna lauany ruaky sax  001"""
            if probabilities.shape[1] == 1:
                roc_auc = float(roc_auc_score(y_test_binarized, probabilities, multi_class="ovr"))
            else:
                """ har classak barabr daka ba classakany tr"""
                roc_auc = float(roc_auc_score(y_test_binarized, probabilities, multi_class="ovr", average="macro"))
                """ averagey yaksan ba ham model bda ba gwerey codaka"""
        except Exception:
            roc_auc = None
            """ mission probley wrong shap"""

    result.roc_auc = roc_auc
    """ har modelk chan bash polakan jia dakata"""
    return result, predictions, probabilities





def build_cnn_model(input_shape: tuple[int, int, int], class_count: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(class_count, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mango_leaf_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

""" auayan rastauy peshbyny poly naxoshy dakat"""
def train_cnn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 10):
    """Train CNN model on raw image arrays (shape: N, H, W, 3).
    
    X_train, X_test should be uint8 images of shape (N, 224, 224, 3).
    Returns (model, history) tuple.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from .config import IMAGE_SIZE, BATCH_SIZE
    
    # Images should already be (N, H, W, 3) RGB uint8
    # Build model with 3-channel input
    class_count = len(np.unique(y_train))
    model = build_cnn_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), class_count=class_count)
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    """✅ بۆچی زیادکردن گرنگە

وێنەی یەک گەڵا دەتوانێت ببێتە چەندین وەشانی:

سووڕاوە
گۆڕدرا
زووم کراوە
وەرگەڕا

سی ئێن ئێن بەهێز دەبێت"""
    
    #  contorly zyraky taybat lakaty rahenan da
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]
    
    # Train - images are scaled by Rescaling layer inside the model
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_test / 255.0, y_test),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    """ go back best model weight"""
    
    return model, history
