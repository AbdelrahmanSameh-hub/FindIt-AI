"""
FindIt AI - Synthetic Campus Lost & Found Dataset
Contains realistic lost items with descriptions (Arabic & English),
synthetic image pixel samples, locations, and timestamps.
"""

from typing import List, Dict, Any, Tuple
import math

# Campus locations and pairwise walking distances (in meters)
CAMPUS_DISTANCES: Dict[Tuple[str, str], float] = {
    ('Library', 'Cafeteria'): 120.0,
    ('Library', 'Engineering_Hall'): 250.0,
    ('Library', 'Bus_Station'): 400.0,
    ('Library', 'Gym'): 550.0,
    ('Cafeteria', 'Engineering_Hall'): 180.0,
    ('Cafeteria', 'Bus_Station'): 300.0,
    ('Cafeteria', 'Gym'): 450.0,
    ('Engineering_Hall', 'Bus_Station'): 220.0,
    ('Engineering_Hall', 'Gym'): 380.0,
    ('Bus_Station', 'Gym'): 600.0,
}

def generate_synthetic_image(dominant_rgb: Tuple[int, int, int], pattern_type: str = 'solid') -> Tuple[List[Tuple[int, int, int]], List[List[float]]]:
    """
    Generates synthetic pixel data:
    1. RGB pixel list for Color Histogram Matching
    2. 32x32 Grayscale grid for pHash DCT extraction
    """
    dr, dg, db = dominant_rgb
    pixels_rgb = []
    gray_grid = [[0.0 for _ in range(32)] for _ in range(32)]

    base_gray = 0.299 * dr + 0.587 * dg + 0.114 * db

    for r in range(32):
        for c in range(32):
            # Apply geometric pattern based on item shape
            intensity_mod = 1.0
            if pattern_type == 'circle':  # e.g., earbuds / coins / watch
                dist = math.sqrt((r - 16)**2 + (c - 16)**2)
                intensity_mod = 1.3 if dist < 10 else 0.4
            elif pattern_type == 'rect':   # e.g., wallet / phone / card
                intensity_mod = 1.2 if (6 < r < 26 and 8 < c < 24) else 0.5
            elif pattern_type == 'stripes':# e.g., backpack / keys
                intensity_mod = 1.25 if (r % 6 < 3) else 0.75

            # Add subtle color noise
            pr = max(0, min(255, int(dr * intensity_mod)))
            pg = max(0, min(255, int(dg * intensity_mod)))
            pb = max(0, min(255, int(db * intensity_mod)))

            pixels_rgb.append((pr, pg, pb))
            gray_grid[r][c] = max(0.0, min(255.0, base_gray * intensity_mod))

    return pixels_rgb, gray_grid


# Database of "FOUND" reports logged by security or students
FOUND_DATABASE: List[Dict[str, Any]] = [
    {
        'id': 'F-101',
        'title': 'سماعة أذن لاسلكية سوداء مع علبة الشحن',
        'title_en': 'Black wireless earbuds with charging case',
        'category': 'Electronics',
        'location': 'Library',
        'location_ar': 'المكتبة المركزية - الطابق الثاني',
        'timestamp': 10.0,  # Hours from baseline
        'description': 'عُثر على علبة سماعة بلوتوث صغيرة لونها أسود ماركة أنكر Anker بجانب طاولة القراءة.',
        'color': (25, 25, 28),     # Matte black
        'pattern': 'circle',
        'confidential_note': 'محفوظة في أمانات المكتبة، يلزم إدخال رمز الاقتران أو الفاتورة للاستلام.'
    },
    {
        'id': 'F-102',
        'title': 'محفظة جلدية بنية داكنة',
        'title_en': 'Dark brown leather wallet',
        'category': 'Personal Accessories',
        'location': 'Cafeteria',
        'location_ar': 'كافتيريا كلية العلوم',
        'timestamp': 14.5,
        'description': 'محفظة رجالية جلد طبيعي بني تحتوي على بعض النقود وبطاقة مواصلات بدون هوية ظاهرة.',
        'color': (90, 50, 30),     # Dark brown
        'pattern': 'rect',
        'confidential_note': 'يتم التحقق بذكر المبلغ التقريبي أو ماركة المحفظة.'
    },
    {
        'id': 'F-103',
        'title': 'حاسبة بيانية كاسيو رمادية',
        'title_en': 'Casio graphing calculator silver gray',
        'category': 'Stationery',
        'location': 'Engineering_Hall',
        'location_ar': 'مدرج كلية الهندسة 3',
        'timestamp': 8.0,
        'description': 'آلة حاسبة علمية كاسيو رمادية اللون fx-991ARX نسيت على أحد المقاعد.',
        'color': (160, 165, 175),  # Silver Gray
        'pattern': 'rect',
        'confidential_note': 'يوجد خدش بسيط على غطاء البطارية.'
    },
    {
        'id': 'F-104',
        'title': 'حقيبة ظهر زرقاء غامقة نايكي',
        'title_en': 'Navy blue Nike backpack',
        'category': 'Bags',
        'location': 'Bus_Station',
        'location_ar': 'موقف الحافلات رقم 2',
        'timestamp': 16.0,
        'description': 'شنطة ظهر رياضية كحلية نايكي تحتوي على دفاتر جامعية وزجاجة ماء معدنية.',
        'color': (15, 35, 95),     # Navy blue
        'pattern': 'stripes',
        'confidential_note': 'اسم الطالب مكتوب داخل الدفتر الأول.'
    },
    {
        'id': 'F-105',
        'title': 'ميدالية مفاتيح سيارة مع شريط أحمر',
        'title_en': 'Car keys with red lanyard',
        'category': 'Keys',
        'location': 'Gym',
        'location_ar': 'مدخل الصالة الرياضية',
        'timestamp': 24.0,
        'description': 'مفتاح سيارة ريموت مع ميدالية قماشية حمراء كتب عليها Remove Before Flight.',
        'color': (200, 20, 30),    # Red lanyard
        'pattern': 'stripes',
        'confidential_note': 'نوع السيارة تويوتا.'
    },
    {
        'id': 'F-106',
        'title': 'سماعة سلكية بيضاء أبل ايربودز',
        'title_en': 'White wired Apple EarPods',
        'category': 'Electronics',
        'location': 'Library',
        'location_ar': 'المكتبة - معمل الحاسوب',
        'timestamp': 30.0,
        'description': 'سماعة ايفون سلكية بيضاء لايتنينج متروكة بجانب شاشة الكمبيوتر.',
        'color': (240, 240, 245),  # White
        'pattern': 'circle',
        'confidential_note': 'ملفوفة بسلك مطاطي أصفر.'
    },
    {
        'id': 'F-107',
        'title': 'نظارة شمسية سوداء بإطار معدني',
        'title_en': 'Black sunglasses with metal frame',
        'category': 'Personal Accessories',
        'location': 'Cafeteria',
        'location_ar': 'الجلسات الخارجية للكافتيريا',
        'timestamp': 48.0,
        'description': 'نظارة شمسية ماركة ريبان بإطار أسود وعدسات داكنة في حافظة سوداء.',
        'color': (30, 30, 35),     # Black
        'pattern': 'rect',
        'confidential_note': 'الحافظة بداخلها منديل تنظيف برتقالي.'
    }
]

# Enrich each found item with synthetic pixel grids & histograms
for item in FOUND_DATABASE:
    pixels, gray_grid = generate_synthetic_image(item['color'], item['pattern'])
    item['pixels_rgb'] = pixels
    item['gray_grid'] = gray_grid


# Ground-truth test queries for system evaluation (Precision@K, Recall@K, MRR)
EVALUATION_QUERIES = [
    {
        'query_id': 'Q-01',
        'lost_description': 'فقدت سماعة سوداء بلوتوث أنكر صغيرة داخل علبتها',
        'target_id': 'F-101',
        'lost_location': 'Library',
        'lost_timestamp': 11.0,
        'lost_color': (22, 22, 26),
        'pattern': 'circle'
    },
    {
        'query_id': 'Q-02',
        'lost_description': 'محفظتي ضاعت لونها بني في المطعم او الكافتيريا',
        'target_id': 'F-102',
        'lost_location': 'Cafeteria',
        'lost_timestamp': 15.0,
        'lost_color': (95, 55, 32),
        'pattern': 'rect'
    },
    {
        'query_id': 'Q-03',
        'lost_description': 'الة حاسبة كاسيو رصاصية فضية مفقودة في قاعة الهندسة',
        'target_id': 'F-103',
        'lost_location': 'Engineering_Hall',
        'lost_timestamp': 8.5,
        'lost_color': (155, 160, 170),
        'pattern': 'rect'
    },
    {
        'query_id': 'Q-04',
        'lost_description': 'شنطة نايك لون كحلي او ازرق نسيتها عند محطة الباصات',
        'target_id': 'F-104',
        'lost_location': 'Bus_Station',
        'lost_timestamp': 17.0,
        'lost_color': (20, 40, 100),
        'pattern': 'stripes'
    },
    {
        'query_id': 'Q-05',
        'lost_description': 'مفاتيح سيارتي فيها ميدالية حمراء ضاعت قرب الجيم',
        'target_id': 'F-105',
        'lost_location': 'Gym',
        'lost_timestamp': 23.0,
        'lost_color': (195, 25, 35),
        'pattern': 'stripes'
    },
    {
        'query_id': 'Q-06',
        'lost_description': 'غرض شخصي دائري صغير لونه أبيض تركته في المكتبة',
        'target_id': 'F-106',
        'lost_location': 'Library',
        'lost_timestamp': 29.0,
        'lost_color': (240, 240, 245),
        'pattern': 'circle'
    },
    {
        'query_id': 'Q-07',
        'lost_description': 'نظارة شمسية أنيقة ضاعت عند جلسات الأكل والمطعم',
        'target_id': 'F-107',
        'lost_location': 'Cafeteria',
        'lost_timestamp': 47.0,
        'lost_color': (32, 32, 36),
        'pattern': 'rect'
    },
    {
        'query_id': 'Q-08',
        'lost_description': 'حاجة لونها بني مستطيلة ضاعت مني وقت الغداء',
        'target_id': 'F-102',
        'lost_location': 'Cafeteria',
        'lost_timestamp': 14.0,
        'lost_color': (92, 52, 28),
        'pattern': 'rect'
    }
]
