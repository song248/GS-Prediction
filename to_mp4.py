import ffmpeg

def convert_webm_to_mp4(input_path, output_path):
    try:
        # 원본 FPS 및 비트레이트 유지
        probe = ffmpeg.probe(input_path)
        video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
        fps = eval(video_stream["r_frame_rate"])  # 프레임 속도 가져오기
        bitrate = video_stream.get("bit_rate", "1000k")  # 기본값 1000k 설정

        (
            ffmpeg
            .input(input_path, fflags='+genpts')  # 정확한 타임스탬프 생성
            .output(output_path,
                    vcodec='libx264',
                    acodec='aac',
                    vf="scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    r=fps,  # 프레임 속도 유지
                    **{"b:v": bitrate},  # 비트레이트 유지
                    preset="veryfast",  # 인코딩 속도 최적화
                    g=1,  # 키 프레임 강제 삽입
                    vsync=0)  # 원본 프레임 유지
            .run(overwrite_output=True)
        )
        print(f"변환 완료: {output_path}")
    except Exception as e:
        print(f"오류 발생: {e}")

# 사용 예제
convert_webm_to_mp4("theft_devmacs.webm", "theft_devmacs.mp4")
