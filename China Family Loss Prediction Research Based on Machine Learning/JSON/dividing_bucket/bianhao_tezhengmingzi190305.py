#! /usr/bin/env python
# -*- coding: utf-8 -*-
from pandas.io.stata import StataReader
import json
# file_name = '/home/chilewang/Desktop/2018zhuizong/2010/2010_adu_dta/2010adult_072016.dta'
# stata_data = StataReader(file_name, convert_categoricals=False)
# columns_list = list(stata_data.read().keys())
# print(columns_list[:300])

# important_feature = {
#     'pid': '个人id',
#     'fid': '家庭样本代码'
# }

provcd = {  # 特征含义
'meaning':'省国标码',
'11': '北京市',
'12': '天津市',
'13': '河北省',
'14': '山西省',
'21': '辽宁省',
'22': '吉林省',
'23': '黑龙江省',
'31': '上海市',
'32': '江苏省',
'33': '浙江省',
'34': '安徽省',
'35': '福建省',
'36': '江西省',
'37': '山东省',
'41': '河南省',
'42': '湖北省',
'43': '湖南省',
'44': '广东省',
'45': '广西壮族自治区',
'50': '重庆市',
'51': '四川省',
'52': '贵州省',
'53': '云南省',
'61': '陕西省',
'62': '甘肃省',
'-9': '缺失',
'65': '新疆维吾尔自治区',
'46': '海南省',
'15': '内蒙古自治区',
'64': '宁夏回族自治区',
'63': '青海省',
}


urban = {
# 特征含义

'meaning':
 '城乡分类变量',

'-9': '未知',

'0': '乡村',


'1': '城市',



}

gender = {  # 特征含义
'meaning':'性别',
'0': '女',
'1': '男',
'-30': '未知',

}



qd3 = {  # 特征含义adult          不同
'meaning':'您现在在上学吗',
'0': '否',
'1': '是',
'-30': '缺失',
'-8': '不适用',

}

qe1_best = {  # 特征含义adult     不同
'meaning':'您现在的婚姻状态的修正',
'-1': '不知道',
'1': '未婚',
'2': '在婚（有配偶',
'3': '同居',
'4': '离婚',
'5': '丧偶',
'-30': '缺失',
}


qe507y = {  # 特征含义adult       不同
'meaning':'您刚过世的配偶去世的日期是（年）',
'-8': '不适用',
#具体年份
'4': '1950年以前',
'5': '1950-1959年',
'6': '1960-1969年',
'7': '1970-1979年',
'8': '1980-1989年',
'9': '1990-1999年',
'10': '2000年以后',
'-30': '缺失',
}

qe211y= {  # 特征含义adult     不同
'meaning':'您现在的配偶的出生年月（年）',
'-8': '不适用',
'4': '1950年以前',
'5': '1950-1959年',
'6': '1960-1969年',
'7': '1970-1979年',
'8': '1980-1989年',
'9': '1990-1999年',
'10': '2000年以后',
'-30': '缺失',


#具体年份

}

qp3= {  # 特征含义adult
'meaning':'您认为自己的健康状况如何',
'1': '健康',
'2': '一般',
'3': '比较不健康',
'4': '不健康',
'5': '非常不健康',
'-30': '缺失',

}#有误，是否为婚姻状况

wm103= {  # 特征含义 child
'meaning':'归根结底，我认为自己是一个失败者',
'-8': '不适用',
'1': '十分不同意',
'2': '不同意',
'3': '同意',
'-30': '缺失',
}

wn2= {  # 特征含义child
'meaning':'上个月，你和父母大概争吵了多少次',
'-30': '缺失',
'-8': '不适用',
'0': '0次',
#数值 0，1，3，10，30等

}

tb2_a_p= {  # 特征含义famcongf
'meaning':'个人性别',
'1': '男',
'0': '女',
'-30': '缺失',
}

birthy_best= {  # 特征含义famcongf
'meaning':'出生日期（年）-修正',
'-1': '不知道',
'4': '1950年以前',
'5': '1950-1959年',
'6': '1960-1969年',
'7': '1970-1979年',
'8': '1980-1989年',
'9': '1990-1999年',
'10': '2000年以后',
'-30': '缺失',

#数值 1950  -  200x

}

alive_a_p= {  # 特征含义famcongf 全部是是
'meaning':'2010年个人是否健在',
'1': '是',
'-30': '缺失',

}



tb3_a_p= {  # 特征含义
'meaning':'个人婚姻状况',
'1': '未婚',
'2': '在婚（有配偶',
'3': '同居',
'4': '离婚',
'5': '丧偶',
'-30': '缺失',


}


tb4_a_p= {  # 特征含义
'meaning':'个人最高学历',
'-8': '不适用',
'-1': '不知道',
'1': '文盲/半文盲',
'2': '小学',
'3': '初中',
'4': '高中/中专/技',
'5': '大专',
'6': '大学本科',
'7': '硕士',
'-30': '缺失',
}


alive_a_f= {  # 特征含义
'meaning':'2010年父亲是否健在',
'-8': '不适用',
'0': '否',
'1': '是',
'-30': '缺失',
}




alive_a_m= {  # 特征含义famcongf adult child
'meaning':'2010年母亲是否健在',
'-8': '不适用',
'0': '否',
'1': '是',
'-30': '缺失',
}

tb6_a_f= {  # 特征含义famcongf adult child
'meaning':'2010年父亲是否住在家中',
'-8': '不适用',
'0': '否',
'1': '是',
'-30': '缺失',
}


tb6_a_m= {  # 特征含义
'meaning':'2010年母亲是否住在家中',
'-8': '不适用',
'0': '否',
'1': '是',
'-30': '缺失',
}



# #数值型，先不填
#
# qa1age= {  # 特征含义
# 'meaning':'年龄',
#
#
# }
#
# qk601= {  # 特征含义
# 'meaning':'您个人的总收入是多少元',
#
#
# }
#
# fe601= {  # 特征含义
# 'meaning':'去年全年出租其他东西的租金总收入为多少(元) ',
#
#
# }
#
#
# fe802= {  # 特征含义
# 'meaning':'您家获得拆迁补偿金总额为多少（元）',
#
#
# }
#
#
# fe903= {  # 特征含义
# 'meaning':'您家获得的征地补偿金总额为多少（元）',
#
#
# }
#
# ff2= {  # 特征含义
# 'meaning':'去年底，您家各种存款余额总计为多少(元)',
#
#
# }
#
#
#
# indinc= {  # 特征含义
# 'meaning':'调整后的家庭人均总收入',
#
#
# }
#
# land_asset= {  # 特征含义
# 'meaning':'去年底，您家各种存款余额总计为多少(元)',
#
#
# }
#
#
# total_asset= {  # 特征含义
# 'meaning':'土地价值-元',
#
# }
#
#
# expense= {  # 特征含义
# 'meaning':'家庭总支出',
#
# }
#
# familysize= {  # 特征含义
# 'meaning':'家庭规模',
#
# }
#
# fproperty= {  # 特征含义
# 'meaning':'财产性收入',
#
# }




# adult_2010_important_dict = {  # 特崢　－　特崢含義
#
# }





provcd_json = json.dumps(provcd)
with open('provcd.json', 'w', encoding='UTF-8') as fw:
    fw.write(provcd_json )

urban_json = json.dumps(urban)
with open('urban.json', 'w', encoding='UTF-8') as fw:
    fw.write(urban_json )

gender_json = json.dumps(gender)
with open('gender.json', 'w', encoding='UTF-8') as fw:
    fw.write(gender_json )

qd3_json = json.dumps(qd3)
with open('qd3.json', 'w', encoding='UTF-8') as fw:
    fw.write(qd3_json )

qe1_best_json = json.dumps(qe1_best)
with open('qe1_best.json', 'w', encoding='UTF-8') as fw:
    fw.write(qe1_best_json )

qe507y_json = json.dumps(qe507y)
with open('qe507y.json', 'w', encoding='UTF-8') as fw:
    fw.write(qe507y_json )

qe211y_json = json.dumps(qe211y)
with open('qe211y.json', 'w', encoding='UTF-8') as fw:
    fw.write(qe211y_json )

qp3_json = json.dumps(qp3)
with open('qp3.json', 'w', encoding='UTF-8') as fw:
    fw.write(qp3_json )

wm103_json = json.dumps(wm103)
with open('wm103.json', 'w', encoding='UTF-8') as fw:
    fw.write(wm103_json)

wn2_json = json.dumps(wn2)
with open('wn2.json', 'w', encoding='UTF-8') as fw:
    fw.write(wn2_json)

birthy_best_json = json.dumps(birthy_best)
with open('birthy_best.json', 'w', encoding='UTF-8') as fw:
    fw.write(birthy_best_json)

alive_a_p_json = json.dumps(alive_a_p)
with open('alive_a_p.json', 'w', encoding='UTF-8') as fw:
    fw.write(alive_a_p_json)

tb3_a_p_json = json.dumps(tb3_a_p)#个人婚姻状况
with open('tb3_a_p.json', 'w', encoding='UTF-8') as fw:
    fw.write(tb3_a_p_json)

tb4_a_p_json = json.dumps(tb4_a_p)#个人最高学历
with open('tb4_a_p.json', 'w', encoding='UTF-8') as fw:
    fw.write(tb4_a_p_json)

alive_a_f_json = json.dumps(alive_a_f)#父亲是否健在
with open('alive_a_f.json', 'w', encoding='UTF-8') as fw:
    fw.write(alive_a_f_json)

alive_a_m_json = json.dumps(alive_a_m)
with open('alive_a_m.json', 'w', encoding='UTF-8') as fw:
    fw.write(alive_a_m_json)

tb6_a_f_json = json.dumps(tb6_a_f)
with open('tb6_a_f.json', 'w', encoding='UTF-8') as fw:
    fw.write(tb6_a_f_json)

tb6_a_m_json = json.dumps(tb6_a_m)
with open('tb6_a_m.json', 'w', encoding='UTF-8') as fw:
    fw.write(tb6_a_m_json)


# with open('important_feature.json', 'r', encoding='UTF-8') as fr:
#     important_feature = json.load(fr)
#     print(important_feature)