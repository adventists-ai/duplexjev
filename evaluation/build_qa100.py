#!/usr/bin/env python3
"""qa100：基础逻辑 + 简单事实问答，100 题（中 50 / 英 50，各含逻辑 25、事实 25）。

每题 4 选项，第一个选项是正确答案；落盘时按题号把正确答案轮转到 A/B/C/D（各 25 次），
其余选项顺序由题号哈希确定，排除位置捷径。语音只念题干，选项以文字给出。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

# (lang, category, question, [gold, distractor, distractor, distractor])
RAW = [
    # ---------- 中文 · 逻辑 ----------
    ("zh", "logic", "小明比小红高，小红比小刚高，三个人里谁最矮？", ["小刚", "小明", "小红", "三个人一样高"]),
    ("zh", "logic", "今天是星期三，再过三天是星期几？", ["星期六", "星期五", "星期日", "星期一"]),
    ("zh", "logic", "先算三加五，再把结果乘以二，最后得到多少？", ["十六", "十一", "十三", "十"]),
    ("zh", "logic", "所有的猫都是动物，咪咪是一只猫，那么咪咪是什么？", ["动物", "植物", "机器", "不能确定"]),
    ("zh", "logic", "如果下雨，地面就会湿。今天地面是干的，那么今天下雨了吗？", ["没有下雨", "下雨了", "下了大雨", "无法判断"]),
    ("zh", "logic", "十二个苹果平均分给四个人，每人分到几个？", ["三个", "四个", "六个", "两个"]),
    ("zh", "logic", "二、四、六、八，下一个数是多少？", ["十", "九", "十二", "十六"]),
    ("zh", "logic", "小李早上八点出门，走了四十五分钟到公司，他几点到公司？", ["八点四十五分", "九点整", "八点半", "九点一刻"]),
    ("zh", "logic", "一张桌子有四条腿，三张桌子一共有几条腿？", ["十二条", "七条", "十条", "十六条"]),
    ("zh", "logic", "妈妈的妈妈应该叫什么？", ["外婆", "奶奶", "姑姑", "阿姨"]),
    ("zh", "logic", "张三站在李四左边，李四站在王五左边，谁站在最右边？", ["王五", "张三", "李四", "无法判断"]),
    ("zh", "logic", "我有十块钱，买了一支三块钱的笔，还剩多少钱？", ["七块", "十三块", "六块", "八块"]),
    ("zh", "logic", "甲比乙大两岁，乙今年十岁，甲今年几岁？", ["十二岁", "八岁", "十岁", "二十岁"]),
    ("zh", "logic", "如果明天是星期一，那么昨天是星期几？", ["星期六", "星期日", "星期五", "星期二"]),
    ("zh", "logic", "一百减去三十七等于多少？", ["六十三", "七十三", "六十七", "五十三"]),
    ("zh", "logic", "冰箱里有五个鸡蛋，吃掉两个，又买了六个，现在一共有几个？", ["九个", "十三个", "三个", "十一个"]),
    ("zh", "logic", "一根直的绳子，不折叠，剪三刀，能剪成几段？", ["四段", "三段", "六段", "五段"]),
    ("zh", "logic", "一辆车每小时开六十公里，开了三个小时，一共开了多少公里？", ["一百八十公里", "一百二十公里", "两百公里", "六十三公里"]),
    ("zh", "logic", "小猫、小狗和小鸟里，哪一个会飞？", ["小鸟", "小猫", "小狗", "都不会飞"]),
    ("zh", "logic", "所有学生都戴着帽子，小王没有戴帽子，小王是不是学生？", ["不是学生", "是学生", "一定是老师", "无法判断"]),
    ("zh", "logic", "七的两倍再减去四，结果是多少？", ["十", "十八", "三", "十四"]),
    ("zh", "logic", "一个班有二十个男生和十五个女生，一共有多少人？", ["三十五人", "二十五人", "三十人", "四十人"]),
    ("zh", "logic", "时针指着三，分针指着十二，现在是几点？", ["三点整", "十二点", "十二点十五分", "三点半"]),
    ("zh", "logic", "哥哥比我大三岁，五年以后哥哥比我大几岁？", ["三岁", "八岁", "五岁", "两岁"]),
    ("zh", "logic", "一个正方形被一条对角线分开，会得到几个三角形？", ["两个", "一个", "三个", "四个"]),
    # ---------- 中文 · 事实 ----------
    ("zh", "fact", "中国的首都是哪座城市？", ["北京", "上海", "广州", "南京"]),
    ("zh", "fact", "在标准大气压下，水烧到多少摄氏度会沸腾？", ["一百度", "五十度", "零度", "两百度"]),
    ("zh", "fact", "太阳从哪个方向升起？", ["东方", "西方", "南方", "北方"]),
    ("zh", "fact", "一天有多少个小时？", ["二十四个", "十二个", "三十六个", "四十八个"]),
    ("zh", "fact", "大熊猫最主要的食物是什么？", ["竹子", "鱼", "牛肉", "苹果"]),
    ("zh", "fact", "地球围绕着什么转？", ["太阳", "月亮", "火星", "木星"]),
    ("zh", "fact", "人体最大的器官是什么？", ["皮肤", "心脏", "肝脏", "大脑"]),
    ("zh", "fact", "彩虹通常有几种颜色？", ["七种", "五种", "三种", "十种"]),
    ("zh", "fact", "《红楼梦》的作者是谁？", ["曹雪芹", "罗贯中", "吴承恩", "施耐庵"]),
    ("zh", "fact", "世界上面积最大的海洋是哪一个？", ["太平洋", "大西洋", "印度洋", "北冰洋"]),
    ("zh", "fact", "植物进行光合作用时会释放出哪种气体？", ["氧气", "二氧化碳", "氮气", "氢气"]),
    ("zh", "fact", "一年有几个季节？", ["四个", "三个", "五个", "六个"]),
    ("zh", "fact", "春节是农历的哪一天？", ["正月初一", "八月十五", "五月初五", "腊月初八"]),
    ("zh", "fact", "端午节人们通常吃什么？", ["粽子", "月饼", "汤圆", "元宵"]),
    ("zh", "fact", "光和声音相比，哪个传播得更快？", ["光更快", "声音更快", "一样快", "无法比较"]),
    ("zh", "fact", "珠穆朗玛峰是世界上最高的什么？", ["山峰", "河流", "湖泊", "沙漠"]),
    ("zh", "fact", "冰是水的哪种状态？", ["固态", "液态", "气态", "等离子态"]),
    ("zh", "fact", "长江最后流入哪片海？", ["东海", "南海", "黄海", "渤海"]),
    ("zh", "fact", "企鹅主要生活在地球的哪个地区？", ["南极", "北极", "赤道", "沙漠"]),
    ("zh", "fact", "奥运会一般每隔几年举办一次？", ["四年", "两年", "一年", "五年"]),
    ("zh", "fact", "蜜蜂会酿造什么？", ["蜂蜜", "牛奶", "丝绸", "棉花"]),
    ("zh", "fact", "中国人口最多的民族是哪个？", ["汉族", "回族", "藏族", "蒙古族"]),
    ("zh", "fact", "月亮自己会发光吗？", ["不会，它反射太阳光", "会，它自己发光", "只在晚上发光", "只在满月时发光"]),
    ("zh", "fact", "中秋节人们通常吃什么？", ["月饼", "粽子", "汤圆", "腊八粥"]),
    ("zh", "fact", "地球上哪种动物的脖子最长？", ["长颈鹿", "大象", "骆驼", "马"]),
    # ---------- English · logic ----------
    ("en", "logic", "Tom is taller than Sam, and Sam is taller than Joe. Who is the shortest?", ["Joe", "Tom", "Sam", "They are the same height"]),
    ("en", "logic", "If today is Friday, what day will it be in two days?", ["Sunday", "Saturday", "Monday", "Thursday"]),
    ("en", "logic", "What is seven plus eight?", ["Fifteen", "Fourteen", "Sixteen", "Thirteen"]),
    ("en", "logic", "All roses are flowers, and this plant is a rose. What is this plant?", ["A flower", "A tree", "A vegetable", "It cannot be told"]),
    ("en", "logic", "Twelve eggs are shared equally among three people. How many eggs does each person get?", ["Four", "Three", "Six", "Twelve"]),
    ("en", "logic", "Five, ten, fifteen, twenty. What number comes next?", ["Twenty-five", "Thirty", "Twenty-two", "Forty"]),
    ("en", "logic", "I had nine apples and gave away four. How many apples do I have left?", ["Five", "Four", "Thirteen", "Six"]),
    ("en", "logic", "It is three o'clock now. What time will it be in four hours?", ["Seven o'clock", "Six o'clock", "Eight o'clock", "One o'clock"]),
    ("en", "logic", "Anna is older than Ben, and Ben is older than Cara. Who is the oldest?", ["Anna", "Ben", "Cara", "It cannot be told"]),
    ("en", "logic", "A car has four wheels. How many wheels do five cars have in total?", ["Twenty", "Nine", "Sixteen", "Twenty-five"]),
    ("en", "logic", "What is twelve divided by four?", ["Three", "Four", "Eight", "Two"]),
    ("en", "logic", "If yesterday was Monday, what day is tomorrow?", ["Wednesday", "Tuesday", "Sunday", "Thursday"]),
    ("en", "logic", "My father's brother is my what?", ["Uncle", "Cousin", "Grandfather", "Nephew"]),
    ("en", "logic", "Which is heavier, one kilogram of iron or one kilogram of feathers?", ["They weigh the same", "The iron", "The feathers", "It cannot be told"]),
    ("en", "logic", "All birds have feathers, and a penguin is a bird. Does a penguin have feathers?", ["Yes", "No", "Only in winter", "It cannot be told"]),
    ("en", "logic", "How many minutes are there in two hours?", ["One hundred twenty", "Sixty", "One hundred", "Two hundred"]),
    ("en", "logic", "What is twenty minus six?", ["Fourteen", "Sixteen", "Twelve", "Twenty-six"]),
    ("en", "logic", "A train leaves at nine o'clock and the trip takes three hours. When does it arrive?", ["Twelve o'clock", "Eleven o'clock", "Ten o'clock", "One o'clock"]),
    ("en", "logic", "Lisa has three sisters and no brothers. How many girls are there among the children in her family?", ["Four", "Three", "Five", "Two"]),
    ("en", "logic", "A box is inside a bag, and the bag is inside a car. Where is the box?", ["Inside the car", "On the car roof", "Outside the car", "Under the car"]),
    ("en", "logic", "What is double eleven?", ["Twenty-two", "Twenty-one", "Twelve", "One hundred eleven"]),
    ("en", "logic", "A week has seven days. How many days are there in three weeks?", ["Twenty-one", "Fourteen", "Twenty-eight", "Ten"]),
    ("en", "logic", "No cats can fly, and Tom is a cat. Can Tom fly?", ["No", "Yes", "Sometimes", "It cannot be told"]),
    ("en", "logic", "What is half of fifty?", ["Twenty-five", "Twenty", "Thirty", "One hundred"]),
    ("en", "logic", "If you are in a race and you pass the person in second place, what place are you in now?", ["Second place", "First place", "Third place", "Last place"]),
    # ---------- English · fact ----------
    ("en", "fact", "What is the capital city of France?", ["Paris", "London", "Berlin", "Rome"]),
    ("en", "fact", "At how many degrees Celsius does water freeze?", ["Zero degrees", "One hundred degrees", "Ten degrees", "Fifty degrees"]),
    ("en", "fact", "Which planet is known as the Red Planet?", ["Mars", "Venus", "Jupiter", "Saturn"]),
    ("en", "fact", "How many continents are there on Earth?", ["Seven", "Five", "Six", "Nine"]),
    ("en", "fact", "What is the largest mammal in the world?", ["The blue whale", "The elephant", "The giraffe", "The hippopotamus"]),
    ("en", "fact", "Who wrote Romeo and Juliet?", ["William Shakespeare", "Charles Dickens", "Mark Twain", "Jane Austen"]),
    ("en", "fact", "Which gas do people need to breathe in to stay alive?", ["Oxygen", "Carbon dioxide", "Helium", "Hydrogen"]),
    ("en", "fact", "How many legs does a spider have?", ["Eight", "Six", "Ten", "Four"]),
    ("en", "fact", "Which ocean lies between Europe and North America?", ["The Atlantic Ocean", "The Pacific Ocean", "The Indian Ocean", "The Arctic Ocean"]),
    ("en", "fact", "What is the closest star to the Earth?", ["The Sun", "Sirius", "Polaris", "Vega"]),
    ("en", "fact", "What sweet food do bees make?", ["Honey", "Milk", "Sugar cane", "Chocolate"]),
    ("en", "fact", "What is the tallest animal in the world?", ["The giraffe", "The elephant", "The horse", "The camel"]),
    ("en", "fact", "How many days are there in a leap year?", ["Three hundred sixty-six", "Three hundred sixty-five", "Three hundred sixty-four", "Three hundred sixty"]),
    ("en", "fact", "In which country do kangaroos live in the wild?", ["Australia", "Brazil", "Canada", "India"]),
    ("en", "fact", "Which musical instrument has black and white keys?", ["The piano", "The guitar", "The violin", "The drum"]),
    ("en", "fact", "Who was the first president of the United States?", ["George Washington", "Abraham Lincoln", "Thomas Jefferson", "John Adams"]),
    ("en", "fact", "In which country are the Great Pyramids of Giza?", ["Egypt", "Mexico", "China", "Greece"]),
    ("en", "fact", "What color are most tree leaves in summer?", ["Green", "Blue", "Purple", "White"]),
    ("en", "fact", "How many sides does a triangle have?", ["Three", "Four", "Five", "Six"]),
    ("en", "fact", "What is the largest planet in our solar system?", ["Jupiter", "Earth", "Mars", "Mercury"]),
    ("en", "fact", "In which season do many trees lose their leaves?", ["Autumn", "Spring", "Summer", "The rainy season"]),
    ("en", "fact", "Which organ pumps blood through the human body?", ["The heart", "The lungs", "The liver", "The stomach"]),
    ("en", "fact", "What is the currency of Japan?", ["The yen", "The dollar", "The euro", "The won"]),
    ("en", "fact", "What do caterpillars turn into?", ["Butterflies", "Spiders", "Beetles", "Frogs"]),
    ("en", "fact", "Which animal is known as the king of the jungle?", ["The lion", "The zebra", "The rabbit", "The monkey"]),
]

LETTERS = "ABCD"


def arrange(i: int, qid: str, opts: list[str]) -> tuple[list[str], str]:
    gold, others = opts[0], list(opts[1:])
    h = hashlib.sha256(qid.encode()).digest()
    others.sort(key=lambda o: hashlib.sha256(h + o.encode()).hexdigest())
    pos = i % 4
    ordered = others[:pos] + [gold] + others[pos:]
    return ordered, LETTERS[pos]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "artifacts/qa100/qa100.json"
    assert len(RAW) == 100, len(RAW)
    items = []
    for i, (lang, cat, q, opts) in enumerate(RAW):
        assert len(opts) == 4 and len(set(opts)) == 4, q
        qid = f"qa-{lang}-{cat}-{i:03d}"
        ordered, gold = arrange(i, qid, opts)
        items.append({"id": qid, "lang": lang, "category": cat, "question": q,
                      "options": dict(zip(LETTERS, ordered)), "gold": gold, "gold_text": opts[0]})
    comp = {}
    for it in items:
        k = f"{it['lang']}-{it['category']}"
        comp[k] = comp.get(k, 0) + 1
    gold_hist = {L: sum(it["gold"] == L for it in items) for L in LETTERS}
    doc = {"kind": "qa100", "n": len(items), "composition": comp, "gold_hist": gold_hist,
           "note": "语音只念 question；options 以文字给出。第一个原始选项为正确答案，已轮转到 A-D。",
           "items": items}
    doc["pack_sha256"] = hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(doc, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(out, comp, gold_hist, doc["pack_sha256"][:16])


if __name__ == "__main__":
    main()
