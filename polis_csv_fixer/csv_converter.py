import csv
import sys

def convert_csv(input_file, output_file):
    """
    轉換CSV資料格式
    輸入：原始CSV包含 tid, txt, agree_count, disagree_count, pass_count 等欄位
    輸出：轉換後的CSV包含 comment-id, comment_text, votes, agrees, disagrees 等欄位
    """
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        # 讀取原始CSV
        reader = csv.DictReader(infile)
        
        # 定義輸出欄位
        fieldnames = [
            'comment-id', 'comment_text', 'votes', 'agrees', 'disagrees', 'passes',
            'a-votes', 'a-agree-count', 'a-disagree-count', 'a-pass-count',
            'b-votes', 'b-agree-count', 'b-disagree-count', 'b-pass-count'
        ]
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            # 計算votes (agree_count + disagree_count + pass_count)
            agree_count = int(row.get('agree_count', 0))
            disagree_count = int(row.get('disagree_count', 0))
            pass_count = int(row.get('pass_count', 0))
            votes = agree_count + disagree_count + pass_count
            
            # 創建新行資料
            new_row = {
                'comment-id': row.get('tid', ''),
                'comment_text': row.get('txt', ''),
                'votes': votes,
                'agrees': agree_count,
                'disagrees': disagree_count,
                'passes': pass_count,
                'a-votes': votes,  # 假設a-votes等於總votes
                'a-agree-count': agree_count,
                'a-disagree-count': disagree_count,
                'a-pass-count': pass_count,
                'b-votes': 0,  # 根據範例，b相關欄位設為0
                'b-agree-count': 0,
                'b-disagree-count': 0,
                'b-pass-count': 0
            }
            
            writer.writerow(new_row)

def main():
    if len(sys.argv) != 3:
        print("使用方法: python csv_converter.py <輸入檔案> <輸出檔案>")
        print("例如: python csv_converter.py input.csv output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        convert_csv(input_file, output_file)
        print(f"轉換完成！輸出檔案：{output_file}")
    except FileNotFoundError:
        print(f"錯誤：找不到輸入檔案 {input_file}")
    except Exception as e:
        print(f"轉換過程中發生錯誤：{e}")

if __name__ == "__main__":
    main()
