# 최종 베이스라인 요약

공통 기준선(원논문): MRS-PRIM(α=0.05, min_support=100) → 경험적 Jaccard → average-linkage → N_Memb(5–10)×K 완전탐색, ρ_eff ≥ 0.6 하 ECC 최대.

| Idea | 최종 베이스라인 | 핵심 수치 | 판정 | 근거 |
|---|---|---|---|---|
| 1 수축 | 방법 B(통합박스 내 2차 PRIM) + inner-validation RATIO2 + 관측치 safeguard(≥25, ≥50% 보존) | Concrete N_Memb*=5, K*=650. 방법 A 개선폭 +0.186(v3), 방법 B는 표본 수 안정 | 채택 | `idea1/idea1_final_report.docx` |
| 2 Tversky | Jaccard 유지 | 최적 W_LARGE 0.6–0.8에서 +0.0009. W_LARGE=0에서 ECC 0.97→0.16 | 기각 | `idea2/idea2_final_report.docx` |
| 3 신뢰도 가중 | γ=0 | 표준에서 γ=1이면 ECC −0.0227~−0.0313. CV≈2% | 중단 | `idea3/idea3_final_report.docx` |
| 4 합성 검증 | 3차 X 생성 + 6차 RSM(R²=0.811) + 통제 위상 실험 | N_eff 47.0 vs 46.4(정답 1 vs 3), 개수 일치 0% | 검증틀 확립 | `idea4/idea4_final_report.docx` |
| 5 DRS 사례 | 원논문 프레임워크를 DRS-PRIM 박스에 적용 | N_Memb*=5, K*=265, N_eff=46, ECC=0.9899, ρ_eff=0.621 | 범용성 확인 | `idea5/idea5_dro_concrete.py` |
| 6 목적함수 | 원논문 규칙 유지. 대안: L-curve 엘보우 | 엘보우 무작위 10시드 +0.0014±0.0042, 방식 4 콘크리트 −0.0053 | 원논문 유지 | `idea6/idea6_final_report_v3.docx` |
