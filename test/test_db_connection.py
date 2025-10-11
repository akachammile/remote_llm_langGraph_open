#!/usr/bin/env python3
"""
数据库连接测试脚本
用于测试MySQL数据库连接是否正常
"""

import sys
from app.database.base import engine


def test_database_connection():
    """
    测试数据库连接
    """
    try:
        # 尝试连接到数据库
        connection = engine.connect()
        print("✅ 数据库连接成功!")
        
        # 执行一个简单的查询来验证连接
        result = connection.execute("SELECT 1")
        row = result.fetchone()
        if row and row[0] == 1:
            print("✅ 数据库查询测试通过!")
        else:
            print("❌ 数据库查询返回了意外结果")
            
        # 关闭连接
        connection.close()
        print("✅ 数据库连接测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False


if __name__ == "__main__":
    print("开始测试数据库连接...")
    success = test_database_connection()
    if success:
        print("\n🎉 所有测试通过，数据库连接正常!")
        sys.exit(0)
    else:
        print("\n💥 数据库连接存在问题，请检查配置!")
        sys.exit(1)