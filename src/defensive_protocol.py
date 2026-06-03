"""防御性分析协议 (Defensive Analysis Protocol, DAP)

在每个实验脚本中 import 此模块，自动执行以下检查：
1. 过拟合检测 (train/test ratio)
2. 数据偏移检测 (feature distribution shift across folds)
3. 标签泄漏检测 (target leakage via feature-label correlation)
4. 结果稳定性检测 (cross-seed variance)
5. 信号真实性检测 (permutation baseline)
6. 梯度健康检测 (gradient norm monitoring)

使用方式:
    from defensive_protocol import DefensiveProtocol
    dap = DefensiveProtocol(experiment_name="EXP1")
    dap.check_overfit(train_rmse, test_rmse)
    dap.check_distribution_shift(X_train, X_test)
    dap.check_leakage(X, y)
    dap.check_stability(results_across_seeds)
    dap.check_permutation_baseline(model_rmse, y_test)
    dap.check_gradients(model)
    report = dap.generate_report()
"""
from __future__ import annotations
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class CheckResult:
    name: str
    passed: bool
    metric: float
    threshold: float
    message: str
    severity: str = "warning"  # "warning", "critical", "info"


class DefensiveProtocol:
    """防御性分析协议: 自动检测实验中的常见陷阱"""

    def __init__(self, experiment_name: str, output_dir: Optional[Path] = None):
        self.experiment_name = experiment_name
        self.output_dir = output_dir or Path("实验/results/dap_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checks: list[CheckResult] = []
        self.warnings_count = 0
        self.critical_count = 0

    # =========================================================
    # 检查 1: 过拟合检测
    # =========================================================
    def check_overfit(
        self,
        train_loss: float,
        test_loss: float,
        ratio_warn: float = 1.5,
        ratio_critical: float = 2.5,
    ) -> CheckResult:
        """检测训练/测试损失比值

        阈值:
          ratio < 1.5: 正常
          1.5 <= ratio < 2.5: 警告 (可能过拟合)
          ratio >= 2.5: 严重 (必须回退)
        """
        ratio = train_loss / test_loss if test_loss > 0 else float("inf")
        # ratio < 1 means test better than train (unusual but ok)
        ratio = max(ratio, 1.0 / ratio) if ratio > 0 else float("inf")
        actual_ratio = test_loss / train_loss if train_loss > 0 else float("inf")

        if actual_ratio < ratio_warn:
            result = CheckResult(
                name="overfit_ratio",
                passed=True,
                metric=actual_ratio,
                threshold=ratio_warn,
                message=f"过拟合可控: test/train={actual_ratio:.2f} < {ratio_warn}",
                severity="info",
            )
        elif actual_ratio < ratio_critical:
            result = CheckResult(
                name="overfit_ratio",
                passed=False,
                metric=actual_ratio,
                threshold=ratio_warn,
                message=f"⚠️ 轻度过拟合: test/train={actual_ratio:.2f}, 建议增加正则化",
                severity="warning",
            )
            self.warnings_count += 1
        else:
            result = CheckResult(
                name="overfit_ratio",
                passed=False,
                metric=actual_ratio,
                threshold=ratio_critical,
                message=f"🚨 严重过拟合: test/train={actual_ratio:.2f}, 必须回退!",
                severity="critical",
            )
            self.critical_count += 1

        self.checks.append(result)
        return result

    # =========================================================
    # 检查 2: 数据分布偏移检测
    # =========================================================
    def check_distribution_shift(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        ks_threshold: float = 0.1,
        max_shift_features: int = 3,
    ) -> CheckResult:
        """用 KS 检验检测训练/测试集特征分布偏移

        如果超过 max_shift_features 个特征的 KS p < ks_threshold → 警告
        """
        n_features = X_train.shape[1]
        shifted_features = []

        for j in range(n_features):
            ks_stat, p_val = stats.ks_2samp(X_train[:, j], X_test[:, j])
            if p_val < ks_threshold:
                shifted_features.append((j, ks_stat, p_val))

        n_shifted = len(shifted_features)
        passed = n_shifted <= max_shift_features

        if passed:
            msg = f"分布偏移可控: {n_shifted}/{n_features} 特征有显著偏移"
            severity = "info"
        else:
            top3 = sorted(shifted_features, key=lambda x: x[1], reverse=True)[:3]
            msg = (f"⚠️ 分布偏移: {n_shifted}/{n_features} 特征偏移显著. "
                   f"Top3: {[(f'feat_{t[0]}', f'KS={t[1]:.3f}') for t in top3]}")
            severity = "warning"
            self.warnings_count += 1

        result = CheckResult(
            name="distribution_shift",
            passed=passed,
            metric=n_shifted / n_features,
            threshold=max_shift_features / n_features,
            message=msg,
            severity=severity,
        )
        self.checks.append(result)
        return result

    # =========================================================
    # 检查 3: 标签泄漏检测
    # =========================================================
    def check_leakage(
        self,
        X: np.ndarray,
        y: np.ndarray,
        corr_threshold: float = 0.95,
    ) -> CheckResult:
        """检测特征与标签之间是否存在异常高相关 (可能的数据泄漏)

        如果任何特征与标签的 |r| > corr_threshold → 严重警告
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_features = X.shape[1]
        n_targets = y.shape[1]
        leaked_pairs = []

        for j in range(n_features):
            for k in range(n_targets):
                r, _ = stats.pearsonr(X[:, j], y[:, k])
                if abs(r) > corr_threshold:
                    leaked_pairs.append((j, k, r))

        passed = len(leaked_pairs) == 0

        if passed:
            # 报告最高相关
            max_r = 0
            for j in range(min(n_features, 50)):  # 只检查前50个特征
                for k in range(n_targets):
                    r, _ = stats.pearsonr(X[:, j], y[:, k])
                    max_r = max(max_r, abs(r))
            msg = f"无标签泄漏: 最大 |r|={max_r:.4f} < {corr_threshold}"
            severity = "info"
        else:
            msg = (f"🚨 疑似标签泄漏! {len(leaked_pairs)} 对特征-标签 |r|>{corr_threshold}: "
                   f"{leaked_pairs[:3]}")
            severity = "critical"
            self.critical_count += 1

        result = CheckResult(
            name="label_leakage",
            passed=passed,
            metric=len(leaked_pairs),
            threshold=0,
            message=msg,
            severity=severity,
        )
        self.checks.append(result)
        return result

    # =========================================================
    # 检查 4: 结果稳定性检测
    # =========================================================
    def check_stability(
        self,
        metrics_across_seeds: list[float],
        cv_threshold: float = 0.15,
    ) -> CheckResult:
        """检测跨随机种子的结果稳定性

        如果变异系数 CV > cv_threshold → 警告 (结果不稳定)
        """
        arr = np.array(metrics_across_seeds)
        mean_val = arr.mean()
        std_val = arr.std()
        cv = std_val / (abs(mean_val) + 1e-8)

        passed = cv < cv_threshold

        if passed:
            msg = f"结果稳定: CV={cv:.4f} < {cv_threshold}, mean={mean_val:.4f}±{std_val:.4f}"
            severity = "info"
        else:
            msg = f"⚠️ 结果不稳定: CV={cv:.4f} >= {cv_threshold}, mean={mean_val:.4f}±{std_val:.4f}"
            severity = "warning"
            self.warnings_count += 1

        result = CheckResult(
            name="stability",
            passed=passed,
            metric=cv,
            threshold=cv_threshold,
            message=msg,
            severity=severity,
        )
        self.checks.append(result)
        return result

    # =========================================================
    # 检查 5: 置换基线检测
    # =========================================================
    def check_permutation_baseline(
        self,
        model_metric: float,
        y_test: np.ndarray,
        n_permutations: int = 100,
        metric_type: str = "rmse",
    ) -> CheckResult:
        """用置换标签计算 chance-level baseline

        如果模型指标未显著优于置换基线 → 模型无效
        """
        rng = np.random.default_rng(2026)
        perm_metrics = []

        for _ in range(n_permutations):
            y_perm = rng.permutation(y_test)
            if metric_type == "rmse":
                perm_m = np.sqrt(np.mean((y_test - y_perm) ** 2))
            elif metric_type == "r":
                if y_test.ndim > 1:
                    rs = [stats.pearsonr(y_test[:, k], y_perm[:, k])[0]
                          for k in range(y_test.shape[1])]
                    perm_m = np.mean(rs)
                else:
                    perm_m = stats.pearsonr(y_test, y_perm)[0]
            perm_metrics.append(perm_m)

        perm_arr = np.array(perm_metrics)
        perm_mean = perm_arr.mean()
        perm_std = perm_arr.std()

        if metric_type == "rmse":
            # 模型 RMSE 应该 < 置换 RMSE
            z_score = (perm_mean - model_metric) / (perm_std + 1e-8)
            passed = model_metric < perm_mean - 1.96 * perm_std
            improvement = (perm_mean - model_metric) / perm_mean * 100
        else:
            # 模型 r 应该 > 置换 r
            z_score = (model_metric - perm_mean) / (perm_std + 1e-8)
            passed = model_metric > perm_mean + 1.96 * perm_std
            improvement = model_metric - perm_mean

        if passed:
            msg = (f"模型显著优于置换基线: model={model_metric:.4f}, "
                   f"perm={perm_mean:.4f}±{perm_std:.4f}, z={z_score:.2f}")
            severity = "info"
        else:
            msg = (f"⚠️ 模型未显著优于置换基线: model={model_metric:.4f}, "
                   f"perm={perm_mean:.4f}±{perm_std:.4f}, z={z_score:.2f}")
            severity = "warning"
            self.warnings_count += 1

        result = CheckResult(
            name="permutation_baseline",
            passed=passed,
            metric=z_score,
            threshold=1.96,
            message=msg,
            severity=severity,
        )
        self.checks.append(result)
        return result

    # =========================================================
    # 检查 6: 梯度健康检测
    # =========================================================
    def check_gradients(
        self,
        grad_norms: list[float],
        vanish_threshold: float = 1e-6,
        explode_threshold: float = 100.0,
    ) -> CheckResult:
        """检测梯度消失/爆炸

        grad_norms: 每个 epoch 的梯度 L2 范数列表
        """
        arr = np.array(grad_norms)
        min_grad = arr.min()
        max_grad = arr.max()
        mean_grad = arr.mean()

        if min_grad < vanish_threshold:
            msg = f"🚨 梯度消失: min_grad={min_grad:.2e} < {vanish_threshold}"
            passed = False
            severity = "critical"
            self.critical_count += 1
        elif max_grad > explode_threshold:
            msg = f"🚨 梯度爆炸: max_grad={max_grad:.2e} > {explode_threshold}"
            passed = False
            severity = "critical"
            self.critical_count += 1
        else:
            msg = f"梯度健康: range=[{min_grad:.4f}, {max_grad:.4f}], mean={mean_grad:.4f}"
            passed = True
            severity = "info"

        result = CheckResult(
            name="gradient_health",
            passed=passed,
            metric=mean_grad,
            threshold=explode_threshold,
            message=msg,
            severity=severity,
        )
        self.checks.append(result)
        return result

    # =========================================================
    # 检查 7: 效应量检测
    # =========================================================
    def check_effect_size(
        self,
        model_metric: float,
        baseline_metric: float,
        metric_type: str = "rmse",
        min_improvement: float = 0.02,
    ) -> CheckResult:
        """检测模型相对基线的效应量是否有实际意义

        如果改善幅度 < min_improvement → 警告 (统计显著但无实际意义)
        """
        if metric_type == "rmse":
            improvement = (baseline_metric - model_metric) / baseline_metric
            passed = improvement > min_improvement
            msg_val = f"RMSE 改善 {improvement*100:.2f}%"
        else:
            improvement = model_metric - baseline_metric
            passed = improvement > min_improvement
            msg_val = f"r 改善 {improvement:.4f}"

        if passed:
            msg = f"效应量有实际意义: {msg_val} > {min_improvement*100:.1f}%"
            severity = "info"
        else:
            msg = f"⚠️ 效应量过小: {msg_val} < {min_improvement*100:.1f}%, 可能无实际意义"
            severity = "warning"
            self.warnings_count += 1

        result = CheckResult(
            name="effect_size",
            passed=passed,
            metric=improvement,
            threshold=min_improvement,
            message=msg,
            severity=severity,
        )
        self.checks.append(result)
        return result

    # =========================================================
    # 检查 8: 多重比较校正
    # =========================================================
    def check_multiple_comparisons(
        self,
        p_values: list[float],
        alpha: float = 0.05,
        method: str = "bonferroni",
    ) -> CheckResult:
        """检测多重比较后是否仍然显著

        method: "bonferroni" 或 "bh" (Benjamini-Hochberg)
        """
        n_tests = len(p_values)
        arr = np.array(p_values)

        if method == "bonferroni":
            corrected_alpha = alpha / n_tests
            n_significant = (arr < corrected_alpha).sum()
            msg_method = f"Bonferroni (α={corrected_alpha:.4f})"
        else:  # BH
            sorted_p = np.sort(arr)
            ranks = np.arange(1, n_tests + 1)
            bh_thresholds = ranks / n_tests * alpha
            n_significant = (sorted_p < bh_thresholds).sum()
            msg_method = f"Benjamini-Hochberg (FDR={alpha})"

        passed = n_significant > 0

        if passed:
            msg = f"多重比较后仍有 {n_significant}/{n_tests} 个显著 ({msg_method})"
            severity = "info"
        else:
            msg = f"⚠️ 多重比较校正后无显著结果 ({msg_method}), 原始 p: {arr.tolist()}"
            severity = "warning"
            self.warnings_count += 1

        result = CheckResult(
            name="multiple_comparisons",
            passed=passed,
            metric=n_significant,
            threshold=1,
            message=msg,
            severity=severity,
        )
        self.checks.append(result)
        return result

    # =========================================================
    # 生成报告
    # =========================================================
    def generate_report(self, verbose: bool = True) -> dict:
        """生成防御性分析报告"""
        report = {
            "experiment": self.experiment_name,
            "total_checks": len(self.checks),
            "passed": sum(1 for c in self.checks if c.passed),
            "warnings": self.warnings_count,
            "critical": self.critical_count,
            "verdict": self._verdict(),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "metric": round(c.metric, 6),
                    "threshold": round(c.threshold, 6),
                    "message": c.message,
                    "severity": c.severity,
                }
                for c in self.checks
            ],
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"  防御性分析报告: {self.experiment_name}")
            print(f"{'='*60}")
            for c in self.checks:
                icon = "✓" if c.passed else ("⚠️" if c.severity == "warning" else "🚨")
                print(f"  {icon} [{c.name}] {c.message}")
            print(f"\n  总结: {report['passed']}/{report['total_checks']} 通过, "
                  f"{report['warnings']} 警告, {report['critical']} 严重")
            print(f"  判定: {report['verdict']}")
            print(f"{'='*60}\n")

        # 保存报告
        out_path = self.output_dir / f"DAP_{self.experiment_name}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2,
                      default=lambda x: bool(x) if isinstance(x, (np.bool_,)) else float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

        return report

    def _verdict(self) -> str:
        if self.critical_count > 0:
            return "🚨 HALT: 存在严重问题，必须回退修复后重跑"
        elif self.warnings_count >= 3:
            return "⚠️ CAUTION: 多个警告，建议审查后再继续"
        elif self.warnings_count > 0:
            return "⚠️ PROCEED_WITH_CAUTION: 有轻微问题，可继续但需在论文中声明"
        else:
            return "✓ PASS: 所有检查通过，可安全继续"

    def should_halt(self) -> bool:
        """如果有严重问题，返回 True 表示应该停止"""
        return self.critical_count > 0

    def should_rollback(self) -> bool:
        """如果过拟合严重或标签泄漏，返回 True 表示应该回退"""
        for c in self.checks:
            if c.severity == "critical" and c.name in ("overfit_ratio", "label_leakage"):
                return True
        return False


# =========================================================
# 便捷函数: 一键全检
# =========================================================
def run_full_check(
    experiment_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_loss: float,
    test_loss: float,
    model_metric: float,
    baseline_metric: float,
    metrics_across_seeds: Optional[list[float]] = None,
    grad_norms: Optional[list[float]] = None,
    p_values: Optional[list[float]] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """一键执行全部防御性检查"""
    dap = DefensiveProtocol(experiment_name, output_dir)

    # 1. 过拟合
    dap.check_overfit(train_loss, test_loss)

    # 2. 分布偏移
    dap.check_distribution_shift(X_train, X_test)

    # 3. 标签泄漏
    X_all = np.vstack([X_train, X_test])
    y_all = np.vstack([y_train, y_test]) if y_train.ndim > 1 else np.concatenate([y_train, y_test])
    dap.check_leakage(X_all, y_all)

    # 4. 置换基线
    dap.check_permutation_baseline(model_metric, y_test, metric_type="rmse")

    # 5. 效应量
    dap.check_effect_size(model_metric, baseline_metric, metric_type="rmse")

    # 6. 稳定性 (如果提供)
    if metrics_across_seeds:
        dap.check_stability(metrics_across_seeds)

    # 7. 梯度健康 (如果提供)
    if grad_norms:
        dap.check_gradients(grad_norms)

    # 8. 多重比较 (如果提供)
    if p_values:
        dap.check_multiple_comparisons(p_values)

    return dap.generate_report()


if __name__ == "__main__":
    # 演示
    rng = np.random.default_rng(42)
    X_tr = rng.normal(0, 1, (200, 10))
    X_te = rng.normal(0.1, 1.1, (50, 10))
    y_tr = rng.normal(0, 1, (200, 6))
    y_te = rng.normal(0, 1, (50, 6))

    report = run_full_check(
        experiment_name="demo",
        X_train=X_tr, X_test=X_te,
        y_train=y_tr, y_test=y_te,
        train_loss=0.8, test_loss=1.2,
        model_metric=1.2, baseline_metric=1.3,
        metrics_across_seeds=[1.18, 1.22, 1.20],
        grad_norms=[0.5, 0.8, 0.3, 0.6],
        p_values=[0.01, 0.03, 0.08, 0.12, 0.45, 0.67],
    )
