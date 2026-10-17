"""
FindIt AI - Interactive Demonstration & Case Study
Simulates the real-world campus scenario:
A student lost black earbuds in the university library.
Shows step-by-step algorithm processing, multimodal matching,
and privacy-preserving handover verification.
"""

import sys
from typing import Tuple
from .engine import FindItEngine
from .dataset import FOUND_DATABASE

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def print_banner():
    banner = """
================================================================================
              FindIt AI — Smart Multimodal Lost & Found System
          نظام ذكي للعثور على المفقودات باستخدام الرؤية الحاسوبية واسترجاع المعلومات
================================================================================
"""
    print(banner)


def run_student_scenario():
    print("-" * 80)
    print(" [1] سيناريو عملي (Practical Case Study): بلاغ طالبة عن سماعة مفقودة")
    print("-" * 80)
    print("📍 السيناريو:")
    print("   طالبة فقدت سماعة سوداء في الجامعة، فتدخل المنصة وترفع الوصف والصورة وتحدد المكان والوقت.")
    print("\n📝 بيانات البلاغ المُدخلة من الطالبة:")
    print("   - الوصف النصي:   'فقدت سماعة سوداء بلوتوث أنكر صغيرة داخل علبتها'")
    print("   - المكان المحتمل:  Library (المكتبة المركزية)")
    print("   - الوقت التقريبي:  11.0 ساعة (من بداية اليوم)")
    print("   - اللون البصري:   RGB(22, 22, 26) - أسود مطفي")
    print("   - النمط الهندسي:  Circle (علبة سماعات مستديرة)")

    print("\n⚙️ جاري تشغيل خوارزميات النظام الأربعة (Running 4 Core Algorithms)...")
    print("   1. Okapi BM25 Ranking                -> استرجاع النصوص وبلاغات المعثورات")
    print("   2. Color Histogram Intersection      -> قياس تشابه التوزيع اللوني 3D RGB")
    print("   3. Perceptual Hashing (2D-DCT)       -> البصمة الهيكلية لمقاومة الإضاءة والزوايا")
    print("   4. Spatio-Temporal Reciprocal Rank   -> دمج الترتيب وحساب القرب المكاني والزمني")

    engine = FindItEngine()

    query_text = "فقدت سماعة سوداء بلوتوث أنكر صغيرة داخل علبتها"
    query_color: Tuple[int, int, int] = (22, 22, 26)
    query_pattern = 'circle'
    query_location = 'Library'
    query_timestamp = 11.0

    results = engine.search(
        query_text=query_text,
        query_color=query_color,
        query_pattern=query_pattern,
        query_location=query_location,
        query_timestamp=query_timestamp,
        top_k=3
    )

    print("\n" + "=" * 80)
    print(" 🎯 نتائج البحث وترتيب البلاغات الأكثر تطابقاً (Search Results)")
    print("=" * 80)

    for rank, res in enumerate(results, start=1):
        item_id = res['item_id']
        raw_item = engine.item_dict[item_id]
        print(f"\n[النتيجة #{rank}] {res['title']} ({item_id})")
        print(f"   📊 نسبة التطابق الإجمالي: {res['match_percentage']}% (Score: {res['final_score']})")
        print(f"   📍 مكان العثور: {res['location']}")
        print(f"   🏷️ التصنيف: {res['category']}")
        print(f"   🔍 تفصيل الرتب (Ranks):")
        print(f"      - رتبة البحث النصي (BM25):   المركز #{res['breakdown']['text_rank']}")
        print(f"      - رتبة التشابه اللوني (Color): المركز #{res['breakdown']['color_rank']}")
        print(f"      - رتبة البصمة الهيكلية (pHash):المركز #{res['breakdown']['phash_rank']}")
        print(f"      - معامل القرب المكاني:       {res['breakdown']['spatial_proximity'] * 100:.1f}%")
        print(f"      - معامل التلاشي الزمني:       {res['breakdown']['temporal_decay'] * 100:.1f}%")

    top_hit = results[0]
    print("\n" + "=" * 80)
    print(" 🔐 بروتوكول التحقق الآمن واستلام المفقودات (Privacy-Safe Verification)")
    print("=" * 80)
    print(f"تم اختيار أعلى نتيجة تطابق بنجاح: [{top_hit['title']}]")
    print("⚠️ لحماية الخصوصية ومنع الادعاءات الكاذبة، لا يتم إظهار صورة كاملة أو بيانات حساسة.")
    print("📋 سؤال التحقق الأمني الخاص بأمانات الجامعة:")
    print(f"   '{top_hit['verification_guide']}'")
    print("✅ بعد تقديم الطالبة للرمز السري أو وصف الماركة، يتم اعتماد تسليم الأمانة رسميًا.")
    print("=" * 80)


def interactive_menu():
    engine = FindItEngine()
    while True:
        print("\n" + "-" * 60)
        print("القائمة الرئيسية لمشروع FindIt AI:")
        print("1. تشغيل السيناريو النموذجي (بلاغ الطالبة عن السماعة)")
        print("2. إدخال بلاغ مفقودات جديد والبحث (Search for Lost Item)")
        print("3. ➕ تسجيل غرض جديد تم العثور عليه (Register a Found Item)")
        print("4. استعراض قاعدة بيانات المعثورات الحالية (View Database)")
        print("5. تشغيل التقييم العلمي وخوارزميات الدقة (Precision / Recall / MRR)")
        print("6. خروج (Exit)")
        print("-" * 60)

        choice = input("اختر رقم الخيار [1-6]: ").strip()
        if choice == '1':
            run_student_scenario()
        elif choice == '2':
            print("\n[إدخال بلاغ مفقود جديد]:")
            text = input("الوصف (مثال: محفظة بني ضاعت مني): ").strip()
            if not text:
                text = "محفظة بني"
            loc = input("المكان المحتمل [Library, Cafeteria, Engineering_Hall, Bus_Station, Gym]: ").strip()
            if loc not in ['Library', 'Cafeteria', 'Engineering_Hall', 'Bus_Station', 'Gym']:
                loc = 'Cafeteria'
            color_choice = input("اللون التقريبي [black, brown, blue, red, silver, white]: ").strip().lower()
            color_map = {
                'black': (25, 25, 25),
                'brown': (90, 50, 30),
                'blue': (20, 40, 100),
                'red': (200, 20, 30),
                'silver': (160, 165, 175),
                'white': (240, 240, 245)
            }
            color_rgb = color_map.get(color_choice, (90, 50, 30))

            results = engine.search(
                query_text=text,
                query_color=color_rgb,
                query_location=loc,
                top_k=3
            )
            print("\n--- أهم النتائج المتطابقة ---")
            for r in results:
                print(f"• {r['title']} | نسبة التطابق: {r['match_percentage']}% | المكان: {r['location']}")
        elif choice == '3':
            print("\n[➕ تسجيل غرض تم العثور عليه في الأمانات]:")
            title = input("اسم الغرض (مثال: نظارة شمسية ريبان): ").strip() or "نظارة شمسية"
            desc = input("الوصف الظاهر (مثال: نظارة شمسية سوداء بإطار معدني): ").strip() or title
            cat = input("التصنيف [Electronics, Personal Accessories, Stationery, Bags, Keys]: ").strip() or "Personal Accessories"
            loc = input("مكان العثور عليه [Library, Cafeteria, Engineering_Hall, Bus_Station, Gym]: ").strip() or "Library"
            secret = input("سؤال التحقق الأمني السري (لا يظهر للعامة): ").strip() or "يتم التحقق بذكر الماركة أو العلامة المميزة."
            
            new_item = engine.add_found_item(
                title=title,
                description=desc,
                category=cat,
                location=loc,
                location_ar=loc,
                color_rgb=(30, 30, 30),
                pattern='rect',
                confidential_note=secret
            )
            print(f"\n✅ تم تسجيل الغرض بنجاح برقم المعرف: [{new_item['id']}] وتحديث فهارس BM25 والرؤية الحاسوبية!")
        elif choice == '4':
            print(f"\nقاعدة بيانات المعثورات المسجلة ({len(engine.items)} أغراض):")
            for item in engine.items:
                print(f"- [{item['id']}] {item['title']} ({item.get('location_ar', item['location'])})")
        elif choice == '5':
            from .evaluate import run_benchmark
            run_benchmark()
        elif choice == '6':
            print("شكراً لاستخدام FindIt AI! بالتوفيق.")
            break
        else:
            print("خيار غير صالح، يرجى اختيار رقم من 1 إلى 6.")


if __name__ == '__main__':
    print_banner()
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_menu()
    else:
        run_student_scenario()
        print("\n💡 تلميح: يمكنك تشغيل الوضع التفاعلي الكامل بكتابة:")
        print("   python -m findit_ai.demo --interactive")
