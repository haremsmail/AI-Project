from __future__ import annotations

"""Feature extraction for handcrafted and deep image descriptors."""
""" taybat wmandy wena lagalay bame bar daka """

""" featuer bashtr bang dakat bo codakan amadakrnd amazha krndy jory paktr la codaka"""

from dataclasses import dataclass
""" bakar de bo druskrndy poly paku halgrtny data"""
from typing import Callable
""" bo halgrnty functily la gorawek da """

import cv2
""" brain of project dozynauay wena bryny wena resize krndy leuarakany """
import numpy as np
""" imagakan ba zory dakrean numpy lanau nupy haldagyren"""
from PIL import Image
""" imagy jory pilow goryny qabary image au shtana bakar de bo save image resize krdny image bakar de """
from skimage.feature import graycomatrix, graycoprops
"""واتا:

فەنکشنەکان لە scikit-image هاوردە دەکات.

بۆ شیکاری پێکهاتە بەکاردێت.

زۆر گرنگە بۆ پەڵە/نەخشەکانی نەخۆشی لەسەر گەڵاکان."""

from .config import IMAGE_SIZE

"""nforzen true wata rastauxo byxuenaua"""
@dataclass(frozen=True)

class DeepFeatureExtractor:
    """ story har datayak daka ka lekolyana ba quly la wenakan dakayn"""
    """ polek data denetau daraua ba quly dep learning lau class ekolyanayan lasar dakja"""
    model: object
    """ aua mabasty auaya objecty tedaya ka deep learning modela"""
    preprocess_input: Callable[[np.ndarray], np.ndarray]
    """واتا:

فەنکشنێکی پێش پرۆسێسکردن هەڵدەگرێت.

ئەم فەنکشنە وێنە ئامادە دەکات پێش ناردنی بۆ ناو سی ئێن ئێن.

نموونە:

ئاساییکردنەوەی پێکسڵەکان
بەهاکانی پێوەر
کەناڵەکان ڕێکبخەنەوە"""




""" au functiana krdnauay wena goryny qabara goryny bo zhmara bo auay ai btuany gnearte bka"""
def load_image(path: str, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """ functiala imagge str path war returny ba numpay """
    image = Image.open(path).convert("RGB").resize(image_size)
    """ image dayxata nau variable pathaka dakatau conver data bo rgb resizy bo nmuna agar 1000 ba 20000 by dayka 240 240 """
    return np.asarray(image, dtype=np.uint8)
""" returny data daka ba number ech pixl number auaan dtype  data type from 0 to 255
for exmaple:
    (224,224,3)

Meaning:

224 rows
224 columns
3 colors (RGB)"""


""" foryny rangy wena bo zhmara basha chuncka zor jara naxoshyana ba pey rang dyary dakren"""
def rgb_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """ image warbgra dabashy bka bo 32 groupy returnysh har ba numbera agadarba
    bins peixl value from 0 to 255"""
    features: list[np.ndarray] = []
    for channel_index in range(3):
        """ loop throgh thrree time color channel rgb 0 1 2"""
        channel_hist, _ = np.histogram(image[..., channel_index], bins=bins, range=(0, 256), density=True)
        """ historgramek drust daka bznay  ha chanalek chand pixle mauda runak
        wata tanya yak chanel wardagre agar chandle list =1 daley kaka bas green warbgra"""
        features.append(channel_hist.astype(np.float32))
        """ goryny feature fromat bo  histogramy dahamy"""
        """ har rangek dakata 32 rangy  zhmary jiawa duatr koy hamau  contcatinati daga daykata vectort"""
    return np.concatenate(features)
"""بۆچی گرنگە بۆ نەخۆشی بامێ

گەڵایەکی تەندروست بەزۆری:

زۆربەیان سەوزن

گەڵای نەخۆش لەوانەیە ئەمانەی هەبێت:

پەڵەی زەرد
پەچەی قاوەیی
قاڵبی ڕەش
سەوزی کاڵ بووەوە

ئەم فەنکشنە ئەو گۆڕانکارییە ڕەنگانە دەگرێت."""




"""ئەم فەنکشنە تایبەتمەندییەکانی ڕەنگی HSV لە وێنەی گەڵای بامێ دەردەهێنێت.

هاوشێوەی هیستۆگرامی RGB یە، بەڵام لەبری بەکارهێنانی:

سوور
سەوز
شین

کەڵک وەردەگرێت:

هوێ
تێربوون
بەها

زۆرجار HSV باشترە بۆ دیاریکردنی نەخۆشی."""

"""| نامە | واتا |
| -------- | ---------------------------------------------- |
| هـ | Hue = ڕەنگی ڕاستەقینە (سەوز، زەرد، قاوەیی، سوور) |
| س | تێربوون = هێز/پاکی ڕەنگ |
| V | بەها = ڕووناکی/تاریکی |"""

def hsv_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    """OpenCV to convert image from RGB format to HSV format. 
    wata valuakan dagore bo au rnag"""
    features: list[np.ndarray] = []
    """ zor basudda au functial la dozynaay naxoshy ruak"""
    for channel_index in range(3):
        channel_hist, _ = np.histogram(hsv_image[..., channel_index], bins=bins, range=(0, 256), density=True)
        features.append(channel_hist.astype(np.float32))
    return np.concatenate(features)
"""گەڵایەکی تەندروست: ١.

ڕەنگ = سەوز
تێربوون = بەرزە
بەها = گەشاوە

کۆپانێکی تۆزاوی: ١.

ڕەنگ = کاڵ
تێربوون = نزم
بەها = ناوچەی سپی/ڕووناکی"""




def glcm_texture_features(image: np.ndarray) -> np.ndarray:
    """ gray level concerent matirx"""
    """" grngy  ba rang nada lekolynaua la srushty  naxshy ruy gala dakat
    ئەم فەنکشنە تایبەتمەندییەکانی پێکهاتە لە وێنەی گەڵاکە دەردەهێنێت.

گرنگی بە ڕەنگ نادات.
لێکۆڵینەوە لە نەخشی ڕووی گەڵا دەکات:

زبری
نەرمی و نەرمی
پەڵە پەڵە
پاودەر
پێکهاتەی قاڵب
شێوازی برینەکان

ئەمەش زۆر بەسوودە چونکە زۆرێک لە نەخۆشیەکان پێکهاتە دەگۆڕن نەک تەنها ڕەنگ"""
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    """ convert image rgb lo gray scale"""
    grayscale = cv2.equalizeHist(grayscale)
    """ auash bo bynyn palau au shtanay nau ruakaka"""
    # Optimize: use fewer distances/angles for speed, but keep levels correct for 8-bit images
    """✅ GLCM چییە؟

GLCM = ماتریکسی هاوڕوودانی ئاستی خۆڵەمێشی

پشکنین دەکات:

چەند جار بەهاکانی ڕووناکی پێکسڵ لە تەنیشت یەکەوە دەردەکەون.

ئەمەش یارمەتی پێوانەکردنی پێکهاتەی وێنە دەدات.
ڕووی ئاسایی نەرم

گەڵای نەخۆش: ١.

پەڵەی زبر پەڵەی ناڕێک پێکهاتەی تۆز"""
    glcm = graycomatrix(
        
        grayscale,
        distances=[1],
        angles=[0, np.pi / 2],
        levels=256,
        symmetric=True,
        normed=True,
    )
    """دروستکردنی ماتریکسی GLCM بە بەکارهێنانی scikit-image.

ئەمەش پەیوەندییەکانی پێکسڵی دراوسێ شی دەکاتەوە.
valuy dauam angel goshaykany horiziotna vertialc
symetyrcial wata a-b b-a hamau 0 90 degree"""
    """ datakanay pauandy pekhatak"""
    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    """هەریەکەیان مانای چییە
1. کۆنتراست

جیاوازی بەهێزی ڕووناکی دەپێوێت.

کۆنتراستی بەرز = پەڵە / برینە تیژەکان.

2. نایەکسانی

جیاوازی پێکسڵ دەپێوێت.

پێکهاتەی ناڕێک بەهای زیاتر دەدات.

3. یەکسانی

نەرمی / یەکپارچەیی دەپێوێت.

ڕەنگە گەڵا تەندروستەکان یەکسانی زیاتریان هەبێت.

4. وزە

ڕێزبەندی / نەخشە دووبارەبووەکان دەپێوێت.

5. پەیوەندی

پەیوەندی نێوان پێکسڵەکانی دراوسێ دەپێوێت.

6. ASM

ساتەوەختی دووەمی گۆشەیی.

یەکپارچەیی پێکهاتە دەپێوێت."""
    return np.array([graycoprops(glcm, prop).mean() for prop in properties], dtype=np.float32)
"""[2.45, 1.82, 0.76, 0.31, 0.88, 0.09] means mayby aoutput """





"""ئەم فەنکشنە تایبەتمەندییەکانی شێوە لە وێنەی گەڵای بامێ دەردەهێنێت.

لێکۆڵینەوە لە ئەندازەیی / هێڵکاری گەڵا دەکات:

قەبارە
درێژی لێوارەکان
گوڵەبەڕۆژە
پانی vs بەرزی
شێوەی چەند پڕە

ئەمەش یارمەتیدەرە چونکە هەندێک نەخۆشی گەڵاکان دەشێوێنن، شوێنەکان بچووک دەبنەوە، کون دروست دەکەن، لوول، لێوارەکانی تێکدەچن."""



def shape_features(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    """ conver image rgb to gray scale"""
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    """ bashtr pishan danu pakrdnauy galakan pesh auay lekolyana lasary bka"""

    """ convert image to  واتا:

گۆڕینی وێنە بۆ ماسکی ڕەش/سپی.

گەڵا دەبێتە:

شتێکی سپی

باکگراوند دەبێتە:

ڕەش
شێوازی ئۆتسۆ:

بە شێوەیەکی ئۆتۆماتیکی باشترین ئاست هەڵدەبژێرێت."""
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    """واتا:

هێڵکاری/سنووری شتەکان بدۆزەرەوە.

کۆنتور = شێوەی لێواری گەڵا."""
    if not contours:
        """ agar hich galayak nadozrya"""
        return np.zeros(5, dtype=np.float32)
    """ agar contours nabu return 5 sfr eakat"""

    largest_contour = max(contours, key=cv2.contourArea)
    """ auayan gawratryn gala haldabzher"""
    """noise psht gue daxa gawratryn gala wardgare wata tanya yak gala warbgra"""
    area = float(cv2.contourArea(largest_contour))
    """ large leave large area"""
    """واتا:

درێژی سنوور لە دەوری گەڵا حیساب بکە.

وەک پێوانەکردنی لێوار بە تار.

لێوارە تێکچووەکانی نەخۆشی لەوانەیە دەوری زیاد بکەن."""
    perimeter = float(cv2.arcLength(largest_contour, True))
    """ bo peana borderaka"""

    x, y, width, height = cv2.boundingRect(largest_contour)
    """ wata goshjay chourgosha ladaruy gala drust daka"""
    bounding_area = float(width * height) if width and height else 1.0
    hull = cv2.convexHull(largest_contour)
    """ tueklly narm la daury gala drust bka"""
    hull_area = float(cv2.contourArea(hull)) or 1.0
    """ bakar de bo dyary krduny chaqu kun"""
    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter else 0.0
    """واتا:

پێوانە دەکات کە شێوەی چەندە گۆڕاوە.

بازنە = نزیک لە 1.0

گەڵای ناڕێک = بەهای کەمتر."""
    solidity = area / hull_area
    aspect_ratio = width / height if height else 0.0
    """ barauardy galay drezhu bchuk dakta"""
    extent = area / bounding_area
    """ chand goshay chuar gosha ba gala pr dakretaua"""
    return np.array([area, perimeter, circularity, solidity, aspect_ratio + extent], dtype=np.float32)
"""واتا:

گەڕانەوەی 5 تایبەتمەندی شێوە:

ناوچە
دەوری
بازنەیی
ڕەقبوون
ڕێژەی ڕووبەر + ڕادە"""




def handcrafted_features(path: str) -> np.ndarray:

    """ chandyn wenaty taybat mand dakata vector lo auay bakar bet lo model"""
    image = load_image(path)
    """ wenakan hazr daka ba bkarnenay hamu funcatjiol bo aauy modealaja eshy lasar bka"""
    return np.concatenate(
        [
            rgb_histogram(image),
            hsv_histogram(image),
            glcm_texture_features(image),
            shape_features(image),
        ]
    )



def extract_handcrafted_matrix(image_or_paths) -> np.ndarray:
    """ process chand wenakay bkat la jiaty yak wena input image path balma auay tr """
    """دروستکردنی تایبەتمەندی دەستی بۆ:

یەک وێنە
یان
چەندین وێنە

پێش ڕاهێنان یان پێشبینیکردن بەکاردێت."""
    if isinstance(image_or_paths, np.ndarray):
        return np.concatenate([
            rgb_histogram(image_or_paths),
            hsv_histogram(image_or_paths),
            glcm_texture_features(image_or_paths),
            shape_features(image_or_paths),
        ])
    # If it's a list of paths, process all
    return np.vstack([handcrafted_features(path) for path in image_or_paths])
    """ bo drustkrny matrix taypat mand ka wenay zory tya"""
    """ the ouptut is some thing limek this
    [
 [features of image1],
 [features of image2],
 [features of image3]
]
✅ لە یەک ڕستەدا

handcrafted_features() مامەڵە لەگەڵ یەک ڕێڕەوی وێنە دەکات، لە کاتێکدا extract_handcrafted_matrix() مامەڵە لەگەڵ تاکە وێنەی بارکراو یان چەندین وێنە دەکات بۆ ڕاهێنان."""




"""ئەمە ئەرکێکی فێربوونی قووڵی زۆر گرنگە لە پڕۆژەکەتدا.

مۆدێلێکی پێشوەختە ڕاهێنراوی سی ئێن ئێن دروست دەکات کە بەکاردێت بۆ دەرهێنانی تایبەتمەندی وێنەی قووڵ لە وێنەی گەڵای بامێ.

ئەم تایبەتمەندیانە بەزۆری بەهێزترن لە تایبەتمەندییە دەستییەکان.

✅ کارایی سەرەکی

ئەم فەنکشنە بە بەکارهێنانی مۆدێلە بەناوبانگەکانی پێش ڕاهێنراو، دەرهێنەری تایبەتمەندی دروست دەکات:

مۆبایل نێتV2
ڕێسنێت50

پاشان شتێک دەگەڕێنێتەوە کە دەتوانێت وێنەیەکی گەڵای بامێ بگۆڕێت بۆ تایبەتمەندی ژمارەیی زیرەک.

بەکاردێت بۆ:

پۆلێنکردنی نەخۆشی باشتر
فێربوونی گواستنەوە
بەراوردکردنی تایبەتمەندی"""
def build_deep_feature_extractor(backbone: str = "MobileNetV2", image_size: tuple[int, int] = IMAGE_SIZE) -> DeepFeatureExtractor:
    """ ama au modeal bakar nahanen hazra tanya bo wrdbynan"""
    if backbone == "MobileNetV2":
        """ auaayn torcy cnn rahneralu la hazranu wena fer  kraua au   modela"""
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        """ wataw wenaakm bo amada bka wata peraper image daka bo modelaka"""

        model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(*image_size, 3))
        """ avg naxshay tabatmandy dagore bo yak vector"""
        """ wata modelek ka peshtr ferkrau shtakany zor bahes"""
        return DeepFeatureExtractor(model=model, preprocess_input=preprocess_input)

    if backbone == "ResNet50":
        """ auayan la mobie ne bashtra balam xautra"""
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

        model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(*image_size, 3))
        return DeepFeatureExtractor(model=model, preprocess_input=preprocess_input)
    raise ValueError(f"Unsupported backbone: {backbone}")
"""✅ بۆچی گرنگە لە پرۆژەی نەخۆشی مانگۆدا

لەبری ئەوەی بە دەست پێوانە بکرێت:

ڕەنگەکان
پێکهاتە
شێوە

سی ئێن ئێن بە شێوەیەکی ئۆتۆماتیکی فێری نەخشە پێشکەوتووەکان دەبێت:

پەڵەی نەخۆشی
سنوورەکانی برینەکان
پێکهاتەی کۆپانێکی تۆزاوی
خوێنبەرە تووشبووەکان
تێکچوونی شێوەی ئاڵۆز"""



"""✅ کارایی سەرەکی

زۆرێک لە ڕێڕەوی وێنەی گەڵای بامێ وەربگرە → بە وەجبە باریان بکە → بنێرە بۆ ناو CNN → گەڕانەوەی ماتریکسی تایبەتمەندی.

بەکاردێت بۆ:

ڕاهێنانی SVM بە بەکارهێنانی تایبەتمەندی قووڵ
بەراوردکردنی تایبەتمەندییە دەستییەکان vs تایبەتمەندییە قووڵەکان
دەرهێنانی تایبەتمەندی خێرا"""
def extract_deep_features(
    image_paths: list[str],
    extractor: DeepFeatureExtractor,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = 32,
    
) -> np.ndarray:
    """batch_size = 32

Process 32 images at one time.

Faster than one-by-on"""
    arrays: list[np.ndarray] = []
    for start_index in range(0, len(image_paths), batch_size):
        """ labar auay sarat ema cnn bakar denyn by defaul agar nmuna 
         100 sample habe daykata
         Loop through image list in groups of 32.

Example:

If 100 images:

0-31
32-63
64-95
96-99"""
        batch_paths = image_paths[start_index : start_index + batch_size]
        batch_images = []
        for path in batch_paths:
            image = load_image(path, image_size=image_size).astype(np.float32)
            batch_images.append(image)
        batch_array = np.asarray(batch_images, dtype=np.float32)
        batch_array = extractor.preprocess_input(batch_array)
        """ image hazra lo process kren ba cnn"""
        batch_features = extractor.model.predict(batch_array, verbose=0)
        """ SEND IMAGE TO CNN"""
        arrays.append(batch_features)
    return np.vstack(arrays)
"""ئەم فەنکشنە زۆرێک لە وێنەکانی گەڵای مانگۆ دەگۆڕێت بۆ ڤێکتەری تایبەتمەندی بەهێز لەسەر بنەمای سی ئێن ئێن بۆ فێربوونی ئامێر."""


""" cnn baxoy modelyk rahneraua saryr wenakan dakatu dayankata vecto"""

"""Images
↓
load_image()
↓
CNN (MobileNetV2 / ResNet50)
↓
Feature vectors
↓
Classifier (SVM / kNN / etc)
LAST FUNCTION"""