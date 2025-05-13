# utils/bond_model.py

from scipy.optimize import newton

def run(bond_price, coupon_rate, face_value, years_to_maturity):
    """
    Block G1 (Advanced): Tính YTM và ước lượng volatility dựa trên duration × interest rate risk.
    Liên kết với app.py và các block H–J trong mô hình tối ưu danh mục hoàn chỉnh.

    Args:
        bond_price (float): Giá thị trường trái phiếu.
        coupon_rate (float): Tỷ lệ coupon (thập phân, ví dụ 0.08 cho 8%).
        face_value (float): Mệnh giá trái phiếu.
        years_to_maturity (int): Số năm còn lại đến đáo hạn.

    Returns:
        bond_return (float): YTM hàng năm (ví dụ: 0.065).
        bond_volatility (float): Rủi ro trái phiếu ước lượng.
        bond_label (str): Nhãn trái phiếu để đưa vào danh mục.
    """

    def bond_price_function(r):
        coupon_cashflows = [
            (coupon_rate * face_value) / (1 + r) ** t
            for t in range(1, int(years_to_maturity) + 1)
        ]
        principal = face_value / (1 + r) ** years_to_maturity
        return sum(coupon_cashflows) + principal

    def estimate_duration(coupon_rate, face_value, ytm, years):
        """
        Tính Macaulay duration để ước lượng độ rủi ro trái phiếu.
        """
        discounted_cashflows = [
            (coupon_rate * face_value * t) / (1 + ytm) ** t
            for t in range(1, years + 1)
        ]
        principal_term = (face_value * years) / (1 + ytm) ** years
        price = bond_price_function(ytm)
        duration = (sum(discounted_cashflows) + principal_term) / price
        return duration

    try:
        ytm = newton(lambda r: bond_price_function(r) - bond_price, x0=0.05)
        if not (0 < ytm < 1):
            ytm = 0.05
    except Exception:
        ytm = 0.05

    duration = estimate_duration(coupon_rate, face_value, ytm, int(years_to_maturity))

    # Giả định volatility lãi suất chuẩn (theo thị trường VN)
    interest_rate_volatility = 0.02  # 2%/năm
    bond_volatility = duration * interest_rate_volatility

    return round(ytm, 6), round(bond_volatility, 4), "CUSTOM_BOND"