import sqlite3
import os

# 自动指向你在 config 里配置的绝对路径
db_path = r"test.db"

# 确保父目录存在
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# 连接数据库（文件不存在会自动创建）
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 创建工业设备 ERP 库存表
cursor.execute('''
CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    part_number TEXT NOT NULL,
    category TEXT,
    stock_quantity INTEGER,
    unit_price REAL,
    warehouse_location TEXT,
    last_updated TEXT
)
''')

# 清空历史数据（防重复运行）
cursor.execute('DELETE FROM inventory')

# 插入高度契合我们项目背景的测试数据
mock_data = [
    ('HF46F/24-HS1', 'Power Relay', 1500, 35.50, 'Shanghai-A1', '2026-09-02'),
    ('HDR-60-24', 'Din Rail Power Supply', 320, 185.00, 'Shenzhen-B2', '2026-09-02'),
    ('HF32FV-16-12-HLTF(590)', 'Signal Relay', 0, 12.00, 'Guangzhou-C3', '2026-09-01'),
    ('PLC-S7-1200', 'Controller', 45, 1250.00, 'Shanghai-A1', '2026-08-28')
]

cursor.executemany('''
INSERT INTO inventory (part_number, category, stock_quantity, unit_price, warehouse_location, last_updated)
VALUES (?, ?, ?, ?, ?, ?)
''', mock_data)

conn.commit()
conn.close()

print(f"✅ 成功生成测试数据库: {db_path}")
print("📊 包含表: inventory (字段: part_number, stock_quantity, unit_price, warehouse_location等)")