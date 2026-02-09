#!/usr/bin/env python3
"""
修復CSV檔案欄位腳本 (新版，處理pol.is的csv)
自動添加 votes 和 passes 欄位
將 comment-body 重命名為 comment_text
使用標準Python庫，不依賴pandas
"""

import csv
import sys
import os

def fix_csv_columns(input_file, output_file=None):
    """
    修復CSV檔案，添加缺失的欄位
    
    Args:
        input_file (str): 輸入CSV檔案路徑
        output_file (str): 輸出CSV檔案路徑，如果為None則覆蓋原檔案
    """
    
    print(f"正在讀取檔案: {input_file}")
    
    try:
        # 讀取CSV檔案
        rows = []
        with open(input_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            print(f"原始欄位: {fieldnames}")
            
            # 檢查必要欄位是否存在
            required_columns = ['agrees', 'disagrees', 'moderated']
            missing_columns = [col for col in required_columns if col not in fieldnames]
            
            if missing_columns:
                print(f"錯誤: 缺少必要欄位: {missing_columns}")
                return False
            
            # 讀取所有行
            for row in reader:
                rows.append(row)
            
            print(f"成功讀取 {len(rows)} 行資料")
        
        # 處理欄位重命名：comment-body -> comment_text
        if 'comment-body' in fieldnames:
            print("正在重命名欄位: comment-body -> comment_text")
            for row in rows:
                if 'comment-body' in row:
                    row['comment_text'] = row['comment-body']
                    del row['comment-body']
            
            # 更新欄位名稱列表
            fieldnames = [col if col != 'comment-body' else 'comment_text' for col in fieldnames]
            print("欄位重命名完成")
        
        # 分析 moderated 欄位的值
        moderated_counts = {}
        for row in rows:
            moderated_val = row['moderated']
            moderated_counts[moderated_val] = moderated_counts.get(moderated_val, 0) + 1
        
        print(f"moderated 欄位值分佈:")
        for val, count in sorted(moderated_counts.items()):
            print(f"  {val}: {count}")
        
        # 1. 添加 votes 欄位：votes = agrees + disagrees + passes
        print("正在計算 votes 欄位...")
        for row in rows:
            agrees = int(row['agrees'])
            disagrees = int(row['disagrees'])
            # 先計算 passes，因為 votes 需要包含它
            moderated_val = row['moderated']
            if moderated_val == '1':
                passes = 0  # 通過的評論沒有 passes
            elif moderated_val == '-1':
                passes = 0  # 不通過的評論沒有 passes
            elif moderated_val == '0':
                passes = 1  # 棄權的評論計為 1 pass
            else:
                passes = 1  # 其他值，假設為棄權
            
            row['passes'] = passes
            row['votes'] = agrees + disagrees + passes
        
        # 2. 統計 passes 數量（用於顯示統計資訊）
        print("正在統計 passes 數量...")
        passes_count = sum(1 for row in rows if row['passes'] == 1)
        
        # 將新欄位添加到 fieldnames 中
        if 'votes' not in fieldnames:
            fieldnames.append('votes')
        if 'passes' not in fieldnames:
            fieldnames.append('passes')
        
        print(f"更新後的欄位: {fieldnames}")
        
        # 計算統計資訊
        total_votes = sum(int(row['votes']) for row in rows) + passes_count
        votes_range = [int(row['votes']) for row in rows]
        min_votes = min(votes_range)
        max_votes = max(votes_range)
        
        print(f"\n欄位統計:")
        print(f"votes 範圍: {min_votes} - {max_votes}")
        print(f"passes 總數: {passes_count}")
        print(f"總投票數 (votes + passes): {total_votes}")
        
        # 重新排列欄位順序，讓 votes 和 passes 在 agrees/disagrees 附近
        new_fieldnames = [
            'timestamp', 'datetime', 'comment-id', 'author-id',
            'agrees', 'disagrees', 'passes', 'votes', 'moderated', 'comment_text'
        ]
        
        # 只包含存在的欄位
        final_fieldnames = [col for col in new_fieldnames if col in fieldnames]
        # 添加其他可能存在的欄位
        other_columns = [col for col in fieldnames if col not in final_fieldnames]
        final_fieldnames.extend(other_columns)
        
        print(f"最終欄位順序: {final_fieldnames}")
        
        # 決定輸出檔案名稱
        if output_file is None:
            # 在原檔案名稱後加上 _fixed
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_fixed.csv"
        
        # 儲存修復後的檔案
        with open(output_file, 'w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=final_fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n修復完成！輸出檔案: {output_file}")
        
        # 顯示前幾行作為驗證
        print(f"\n前5行資料預覽:")
        for i, row in enumerate(rows[:5]):
            print(f"行 {i+1}: comment-id={row['comment-id']}, agrees={row['agrees']}, disagrees={row['disagrees']}, passes={row['passes']}, votes={row['votes']}, moderated={row['moderated']}")
            if 'comment_text' in row:
                comment_preview = row['comment_text'][:50] + "..." if len(row['comment_text']) > 50 else row['comment_text']
                print(f"      comment_text: {comment_preview}")
        
        return True
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函式"""
    if len(sys.argv) < 2:
        print("使用方法: python3 csv_converter_new.py <輸入CSV檔案> [輸出CSV檔案]")
        print("範例: python3 csv_converter_new.py comments_realdata.csv")
        print("範例: python3 csv_converter_new.py comments_realdata.csv comments_fixed.csv")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"錯誤: 檔案不存在: {input_file}")
        return
    
    success = fix_csv_columns(input_file, output_file)
    
    if success:
        print("\n✅ CSV檔案修復成功！")
    else:
        print("\n❌ CSV檔案修復失敗！")

if __name__ == "__main__":
    main()
