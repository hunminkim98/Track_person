from moviepy.editor import VideoFileClip
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os

class VideoInfo:
    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path)
        self.start_time = tk.StringVar()
        self.end_time = tk.StringVar()
        
    def get_output_path(self):
        """원본 파일 경로에 _trimmed를 추가한 저장 경로 반환"""
        directory = os.path.dirname(self.path)
        filename = os.path.basename(self.path)
        name, ext = os.path.splitext(filename)
        return os.path.join(directory, f"{name}_trimmed{ext}")

def select_videos():
    """여러 비디오 파일을 선택하는 함수"""
    file_paths = filedialog.askopenfilenames(
        title="비디오 파일들 선택",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    return [VideoInfo(path) for path in file_paths]

def trim_video(video_info, progress_var=None, progress_label=None):
    """비디오를 자르는 함수"""
    try:
        # 시간 값 검증
        try:
            start_time = float(video_info.start_time.get())
            end_time = float(video_info.end_time.get())
        except ValueError:
            raise ValueError(f"'{video_info.filename}'의 시간 값이 올바르지 않습니다.")

        # 비디오 파일 로드
        video = VideoFileClip(video_info.path)
        
        # 시간 범위 검증
        if start_time < 0 or end_time > video.duration or start_time >= end_time:
            raise ValueError(
                f"'{video_info.filename}'의 시간 범위가 올바르지 않습니다.\n"
                f"비디오 길이: {video.duration:.1f}초"
            )
        
        # 비디오 자르기
        trimmed_video = video.subclip(start_time, end_time)
        
        # 잘린 비디오 저장
        output_path = video_info.get_output_path()
        trimmed_video.write_videofile(output_path)
        
        # 리소스 해제
        video.close()
        trimmed_video.close()
        
        return True
    except Exception as e:
        messagebox.showerror("에러", str(e))
        return False

def process_videos(videos, progress_var, progress_label):
    """여러 비디오를 처리하는 함수"""
    total_videos = len(videos)
    success_count = 0
    
    for idx, video_info in enumerate(videos, 1):
        # 진행 상태 업데이트
        progress_label.config(text=f"처리 중... ({idx}/{total_videos}): {video_info.filename}")
        progress_var.set((idx - 1) / total_videos * 100)
        
        # 비디오 처리
        if trim_video(video_info, progress_var, progress_label):
            success_count += 1
            
    # 최종 진행 상태 업데이트
    progress_var.set(100)
    progress_label.config(text=f"완료! {success_count}/{total_videos} 개 처리됨")
    
    return success_count

class VideoListFrame(ttk.Frame):
    def __init__(self, master, videos):
        super().__init__(master)
        self.videos = videos
        
        # 스크롤바가 있는 프레임 생성
        self.canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # 헤더 생성
        ttk.Label(self.scrollable_frame, text="파일명", width=40).grid(row=0, column=0, padx=5)
        ttk.Label(self.scrollable_frame, text="시작 시간 (초)", width=15).grid(row=0, column=1, padx=5)
        ttk.Label(self.scrollable_frame, text="종료 시간 (초)", width=15).grid(row=0, column=2, padx=5)

        # 각 비디오별 입력 필드 생성
        for idx, video in enumerate(videos, 1):
            ttk.Label(self.scrollable_frame, text=video.filename).grid(row=idx, column=0, padx=5, pady=2)
            ttk.Entry(self.scrollable_frame, textvariable=video.start_time, width=15).grid(row=idx, column=1, padx=5)
            ttk.Entry(self.scrollable_frame, textvariable=video.end_time, width=15).grid(row=idx, column=2, padx=5)

        # 레이아웃
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

def main():
    # GUI 윈도우 생성
    root = tk.Tk()
    root.title("비디오 배치 편집기")
    root.geometry("800x600")

    def select_and_show_videos():
        # 비디오 파일들 선택
        videos = select_videos()
        if not videos:
            return
        
        # 기존 위젯들 제거
        for widget in root.winfo_children():
            widget.destroy()
        
        # 비디오 목록 표시
        video_list = VideoListFrame(root, videos)
        video_list.pack(fill="both", expand=True, padx=10, pady=10)

        # 진행 상태 표시
        progress_var = tk.DoubleVar()
        progress_label = tk.Label(root, text="대기 중...")
        progress_label.pack(pady=5)
        progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=5)

        def start_processing():
            # 비디오 처리 실행
            success_count = process_videos(videos, progress_var, progress_label)
            
            # 완료 메시지
            messagebox.showinfo("완료", 
                f"배치 처리가 완료되었습니다!\n성공: {success_count}개\n실패: {len(videos) - success_count}개")

        # 실행 버튼
        process_button = tk.Button(root, text="비디오 처리 시작", command=start_processing)
        process_button.pack(pady=10)

    # 초기 화면
    select_button = tk.Button(root, text="비디오 파일 선택", command=select_and_show_videos)
    select_button.pack(expand=True)

    # 도움말 텍스트
    help_text = """
    사용 방법:
    1. '비디오 파일 선택' 버튼을 클릭하여 처리할 비디오들을 선택
    2. 각 비디오의 시작 시간과 종료 시간을 초 단위로 입력
    3. '비디오 처리 시작' 버튼을 클릭
    
    * 저장된 파일은 원본 파일과 같은 위치에 '_trimmed' 접미사가 붙어 저장됩니다.
    """
    tk.Label(root, text=help_text, justify=tk.LEFT).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
