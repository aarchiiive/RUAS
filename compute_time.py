import torch
from torchprofile import profile_macs  # MACs 계산 라이브러리
from model import Network  # 모델 정의 가져오기

# 모델 로드 후 GPU로 이동
model = Network()
model = model.cuda()

# 입력 크기 정의 (예: 720x1080 이미지, 채널 3)
input_size = (720, 1080)
dummy_input = torch.randn(1, 3, *input_size).cuda()

# 초기 더미 실행: 100번 실행하여 GPU 관련 오버헤드 제거
for _ in range(100):
    _ = model(dummy_input)

# 100번의 추론 시간을 기록할 리스트 생성
times = []
for _ in range(100):
    # 이전 GPU 연산이 모두 끝났는지 동기화
    torch.cuda.synchronize()

    # 시작과 종료 이벤트 생성 (시간 측정을 위해 enable_timing=True 설정)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 시작 시간 기록
    start_event.record()
    _ = model(dummy_input)  # 추론 실행
    # 종료 시간 기록
    end_event.record()

    # 모든 GPU 연산이 끝날 때까지 동기화
    torch.cuda.synchronize()
    # 시작과 종료 이벤트 사이의 경과 시간을 밀리초 단위로 측정
    elapsed_time_ms = start_event.elapsed_time(end_event)
    times.append(elapsed_time_ms)

# 100번 실행에 대한 평균 추론 시간 계산
avg_time = sum(times) / len(times)
print("Average inference time over 100 runs: {:.3f} ms".format(avg_time))
