"""Project-wide configuration and paths."""
""" la jiaty auay bo nmuna bley 
desktop/week1/data """
from pathlib import Path
""" handle file folder cleanly"""
""" la jiaty auay hamu katek foldarkan fialkan bangn bkatua confi filekt haya eshakant bo daka
 hamu au reurauana haldagre ka filakany tr bakary dahenen"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
""" folder saraky prozhaka automatricaly ba dast deene"""
DATA_DIR = PROJECT_ROOT / "data"
"""" create path for all data"""
RAW_DATA_DIR = DATA_DIR / "raw"
""" folder contains original data set """
PROCESSED_DATA_DIR = DATA_DIR / "processed"
""" folder bo data gorderauakanu qapara gorauakan 
nmuna wenakay qabary gorawakan"""
MODELS_DIR = PROJECT_ROOT / "models"
""" folder to save training ai models"""
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
""" contains jubtyear notebook
پۆلێنکردنی_نەخۆشی_گەڵا_بامێ.ipynb

فایلەکانی دەفتەر فایلە کارلێککارەکانی پایتۆنن کە بۆ:

نووسینی کۆد لە خانەکاندا
کۆدی جێبەجێکردن هەنگاو بە هەنگاو
نیشاندانی گرافەکان
ڕاهێنانی مۆدێلەکانی AI
تاقیکردنەوەی وێنەکان
ڕوونکردنەوەی ئەنجامەکان"""
RESULTS_DIR = PROJECT_ROOT / "results"
""" au sheuanay ka result teda haldagyre"""

IMAGE_SIZE = (224, 224)
"""  bo modely cnn modley qures au sizea bakjrd deep learning"""
CLASSICAL_IMAGE_SIZE = (128, 128)
"""kNN
SVM
Naive Bayes"""
RANDOM_STATE = 42
""" controly randomness"""
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
""" wata taqy krndua 
Meaning:

20% of data reserved for testing.

If dataset = 4000 images:

3200 training
800 testing"""

BATCH_SIZE = 32
""" la katy bakar henany modely cnn 32 image ba yakaua process bkat memory zory daue"""
EPOCHS = 25
""" 25 teparbuny tawaw lanau data setaka 
 la reagay neural  network hamu  image dabyne"""
KNN_NEIGHBORS = 7
""" wata ba layany kamaua classify 7 wena daak"""
SVM_C = 10.0
""" auaayan svm balam matrsy zyadaroy lasara ka anjamy dat """


"""✅ بۆچی دەفتەر بەکار بهێنین؟

لەبری ئەوەی یەک سکریپتی درێژ بەڕێوەببەیت، دەفتەر ڕێگەت پێدەدات:

خانەی یەکەم:

داتا سێت بارکردن

خانەی دووەم:

وێنەی نمونەیی پیشان بدە

خانەی سێیەم:

مۆدێلی سی ئێن ئێن ڕابهێنە

خانەی چوارەم:

گرافیکی وردبینی پیشان بدە

خانەی پێنجەم:

پێشبینی وێنەی تاقیکردنەوە بکە"""