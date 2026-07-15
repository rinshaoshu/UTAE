import random

# ========== 在这里修改您的设置 ==========
input_file = 'all.txt'           # 输入文件名
ratio1, ratio2, ratio3 = 0.7, 0.1, 0.2  # 三组比例（相加要等于1）
# ======================================

# 读取文件
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 打乱顺序
random.shuffle(lines)

# 按比例分组
total = len(lines)
n1 = int(total * ratio1)
n2 = int(total * ratio2)
# 第三组拿剩下的，保证总数不变
n3 = total - n1 - n2

group1 = lines[:n1]
group2 = lines[n1:n1+n2]
group3 = lines[n1+n2:]

# 输出三个文件
with open('train.txt', 'w', encoding='utf-8') as f:
    f.writelines(group1)

with open('val.txt', 'w', encoding='utf-8') as f:
    f.writelines(group2)

with open('test.txt', 'w', encoding='utf-8') as f:
    f.writelines(group3)

# 显示结果
print(f"总行数: {total}")
print(f"目标比例: {ratio1:.0%} : {ratio2:.0%} : {ratio3:.0%}")
print(f"实际分组: {len(group1)} : {len(group2)} : {len(group3)}")
print(f"实际比例: {len(group1)/total:.1%} : {len(group2)/total:.1%} : {len(group3)/total:.1%}")