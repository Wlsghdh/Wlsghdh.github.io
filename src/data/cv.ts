/**
 * 이력서/포트폴리오 데이터 — 항목만 추가하면 홈과 /cv(Phase 4)에 반영됩니다.
 * 출처: 주진호 — 이력서 / Jin-Ho Ju — CV (2026)
 */

export interface Publication {
  /** [P1]식 라벨 */
  tag: string;
  title: string;
  /** 저자 배열 — 내 이름은 렌더링 시 CAU Blue 볼드 처리 */
  authors: string[];
  venue: string;
  note?: string;
  summary: string;
  links: { label: string; url: string }[];
}

export interface Project {
  title: string;
  period: string;
  subtitle?: string;
  summary: string;
  stack: string[];
  links: { label: string; url: string }[];
}

export const education = [
  {
    period: '2026.07.13 ~ 현재',
    school: '중앙대학교',
    detail: '학부 연구 인턴',
  },
  {
    period: '2021.03 ~ 2027.02 (졸업 예정)',
    school: '수원대학교',
    detail: '데이터과학부 학사과정 · GPA 4.33 / 4.5',
  },
  {
    period: '2019.03 ~ 2021.02',
    school: '성일고등학교',
    detail: '',
  },
];

export const experience = [
  {
    period: '2026.02 ~ 2026.06',
    title: '학부생 조교 — DSML Vibe Coding',
    org: '수원대학교 DSML',
    detail: '학부생 대상 멘토링을 수행했습니다.',
  },
  {
    period: '2025.09 ~ 2025.12',
    title: '연구 인턴 — Hadd Science',
    org: '성균관대학교',
    detail: '홈페이지 관리, 기사 작성, AutoCAD 3D 작업을 수행했습니다.',
  },
  {
    period: '2025.07 ~ 2026.06',
    title: '학부연구생',
    org: '수원대학교 안홍렬 교수 연구실',
    detail:
      '제1저자·공저자 논문으로 이어진 컴퓨터 비전 연구를 수행했습니다. 연구와 병행해 React / Node.js / MongoDB 기반 응용 프로토타입도 구축했습니다.',
  },
  {
    period: '2024.08 ~ 2025.02',
    title: '자율주행팀 인턴',
    org: 'AIMMO Inc.',
    detail:
      '비전 실패 데이터 정제, Labellerr 검수, 학습 데이터 수집, 탐지 실패 센서 그래프 분석을 수행했습니다.',
  },
  {
    period: '2022.01 ~ 2023.07',
    title: '군 복무 — 대한민국 해병대',
    org: '',
    detail: '병역 의무 만기 전역.',
  },
];

export const publications: Publication[] = [
  {
    tag: 'P1',
    title:
      'Why Deep ResNets Train Successfully: Self-Selection of Effective Depth Enabled by Skip Connections',
    authors: ['주진호', '안홍렬'],
    venue: 'arXiv · BMVC 2026 심사 중',
    note: '제1저자',
    summary:
      '잔차 블록마다 학습 가능한 스칼라 α를 도입한 Learnable Residual Scaling(LRS)을 제안 — 깊은 ResNet이 명목 깊이보다 훨씬 작은 effective depth를 스스로 선택함을 입증했습니다(200층 중 active 5~6 블록). ResNet-200/CIFAR-100에서 α < 0.03 블록 제거 시 재학습 없이 정확도 손실 0을 확인했습니다.',
    links: [{ label: 'Code', url: 'https://github.com/Wlsghdh/Learnable_Residual_Scaling' }],
  },
  {
    tag: 'P2',
    title: 'Effects of Generative-AI Augmentation for Small-Sample Industrial Defect Detection',
    authors: ['주진호', '임대윤', '양진우', '안홍렬'],
    venue: 'KIISE 2026',
    note: '제1저자',
    summary:
      '산업 결함 탐지의 데이터 희소성(클래스당 ~20장)에서 의미론적 다양성이 양보다 중요함을 입증 — 전통 증강은 mAP −1.76, Gemini-2.0 기반 생성형 증강은 +1.90, 둘을 조합한 8× 최적점에서 +5.0 mAP를 검출기 계열 전반에서 확인했습니다.',
    links: [{ label: 'Code', url: 'https://github.com/Wlsghdh/VISION-Instance-Seg' }],
  },
  {
    tag: 'P3',
    title:
      'An Integrated Preprocessing Pipeline for Model Performance Comparison on a Multimodal Gas Sensor Dataset',
    authors: ['맹영민', '주진호', '윤재훈', '정우창', '안홍렬'],
    venue: 'KIISE 2025',
    note: '제2저자',
    summary:
      'MultimodalGasData 벤치마크를 위한 4단계 표준 전처리 파이프라인을 설계하고 8개 모델을 공정 비교 — 최고와 최저 간 22.2%p 격차를 도출해 융합 전략이 모델 깊이보다 중요함을 입증했습니다.',
    links: [{ label: 'Code', url: 'https://github.com/Ahn-Laboratory/Gas-Leakage-Detection' }],
  },
];

export const projects: Project[] = [
  {
    title: 'CarNeRF — AI 기반 중고차 딜러 플랫폼',
    period: '2025.12 ~ 2026.07',
    subtitle: '개인/팀 프로젝트 · 공모전',
    summary:
      '판매자의 1분 영상이 FastGS 기반 3D 복원 → YOLOv8 결함 탐지 → LightGBM 가격 예측 → LLM 자연어 검색으로 이어지는 자동화 파이프라인. Vanilla 3DGS 기준 PSNR 31.88, FastGS 이식 후 약 15배 속도 향상.',
    stack: ['FastGS', 'COLMAP', 'SAM', 'YOLOv8', 'LightGBM', 'FastAPI', 'RAG', 'React Native'],
    links: [
      { label: 'Code', url: 'https://github.com/Wlsghdh/CarNeRF' },
      { label: 'Demo', url: 'http://lifeai.suwon.ac.kr:5199' },
    ],
  },
  {
    title: 'ETF with AI — LambdaRank 기반 트레이딩 시스템',
    period: '2026.02 ~ 현재',
    subtitle: '개인/팀 프로젝트 · 진행 중인 공모전',
    summary:
      'LightGBM LambdaRank를 85개 특징 위에 학습하고 TradingView 자동 수집기와 Next.js 대시보드를 통합. 한국투자증권 API 연동 실거래, cron 기반 일일 예측·월간 재학습. 1,000 티커로 확장 중.',
    stack: ['LightGBM LambdaRank', 'FastAPI', 'Next.js', 'Playwright', 'MySQL', 'Docker', 'KIS API'],
    links: [
      { label: 'Code', url: 'https://github.com/Wlsghdh/etf-trading-projects' },
      { label: 'Demo', url: 'https://ahnbi2.suwon.ac.kr/trading' },
    ],
  },
  {
    title: 'JUMP AI 2025 — 제3회 신약 개발 경진대회',
    period: '2025.07 ~ 08',
    subtitle: 'DACON · 1인 참가 · Top 4% (20위 / 1,134팀)',
    summary:
      'SMILES 입력으로 MAP3K5 IC50 활성 예측. 140차원 특징 설계, 5개 베이스 학습기를 Optuna로 튜닝한 뒤 Two-track 앙상블(SLSQP 가중 + Quantile-matching)로 결합.',
    stack: ['RDKit', 'LightGBM', 'XGBoost', 'Optuna', 'scikit-learn'],
    links: [{ label: 'Code', url: 'https://github.com/Wlsghdh/Jump-AI-2025' }],
  },
  {
    title: '퍼스널 컬러 진단 시스템',
    period: '2024.06 ~ 2025.03',
    subtitle: '개인/팀 프로젝트 · 화성시 AI 포럼 초청 강연',
    summary:
      '얼굴 사진으로 퍼스널 컬러를 진단하고 맞춤 영상 콘텐츠를 재생하는 인터랙티브 부스 웹 서비스. 셀럽 레퍼런스 ~5만 장 수집, White-balancing + OpenCV 피부톤 분할, OpenAI 프롬프트 라우팅.',
    stack: ['PyTorch', 'OpenCV', 'OpenAI API', 'AWS', 'Docker', 'React'],
    links: [{ label: 'Code', url: 'https://github.com/Woochang4862/personal-color-app' }],
  },
];

export const awards = [
  { date: '2021 ~ 2026', title: '성적우수장학금', org: '수원대학교 · 성적 기반, 5학기 수혜' },
  { date: '2025.12', title: '장려상', org: '한국정보과학회 (KIISE)' },
  { date: '2025.11', title: '1위 — 주식 차트 데이터 수집 #2', org: '수원대학교 DSML' },
  { date: '2025.10', title: '1위 — 주식 차트 데이터 수집 #1', org: '수원대학교 DSML' },
  { date: '2025.08', title: 'Top 4% · 20위 / 1,134팀 — JUMP AI 2025 신약 개발 경진대회', org: 'DACON' },
  { date: '2024.08', title: '우수상 — AI/Develops', org: '수원대학교 DSML' },
];

export const skills = [
  {
    category: 'Deep Learning',
    items: ['PyTorch', 'PyTorch Lightning', 'LLM Fine-tuning (LoRA/PEFT)', 'DDP', 'Optuna', 'W&B'],
  },
  {
    category: 'Computer Vision',
    items: ['YOLO', 'detectron2', 'mmdetection', 'OpenCV', 'Gaussian Splatting', 'COLMAP', 'SAM'],
  },
  { category: 'NLP & LLM', items: ['HuggingFace', 'RAG', 'Prompt Engineering'] },
  {
    category: 'Engineering',
    items: ['FastAPI', 'SQLAlchemy', 'Next.js', 'Docker', 'MySQL', 'Selenium', 'Playwright'],
  },
  { category: 'Workflow', items: ['Git (PR/Branch/Issue)', 'Linux', 'AWS'] },
];

export const certificates = [
  { date: '2025.02', title: 'Google Data Analytics Professional Certificate', org: 'Coursera × Google' },
  { date: '2025.11 ~ 12', title: '디지털 헬스케어 (입문)', org: 'GUIP 바이오헬스 플랫폼' },
];
