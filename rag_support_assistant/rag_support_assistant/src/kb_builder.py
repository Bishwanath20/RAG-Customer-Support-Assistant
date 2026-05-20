# ═══════════════════════════════════════════════════════════════
# kb_builder.py — Generates the customer support knowledge base PDF
# In production: replace with your actual company PDF document
# ═══════════════════════════════════════════════════════════════

import os
from src.config import DATA_DIR, PDF_PATH


def create_knowledge_base(path: str = PDF_PATH) -> str:
    """
    Build a professional 8-section customer support knowledge base PDF.

    Sections:
        1. Return & Refund Policy
        2. Shipping & Delivery
        3. Account & Password Reset
        4. Payment Methods
        5. Order Cancellation
        6. Product Warranty
        7. Customer Support Contacts
        8. Loyalty Rewards Program

    Returns
    -------
    str : Path to the created PDF file
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import inch
    except ImportError:
        raise ImportError("reportlab not installed. Run: pip install reportlab")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    doc  = SimpleDocTemplate(path, pagesize=A4,
               topMargin=inch, bottomMargin=inch,
               leftMargin=inch, rightMargin=inch)
    h1   = ParagraphStyle("h1",   fontSize=14, fontName="Helvetica-Bold",
                          spaceAfter=8, spaceBefore=16)
    body = ParagraphStyle("body", fontSize=11, fontName="Helvetica",
                          spaceAfter=6, leading=17)
    story = []

    sections = [
        ("1. Return & Refund Policy",
         "Our return policy allows customers to return any product within 30 days of purchase. "
         "To be eligible, the item must be unused, in its original packaging, and accompanied "
         "by the original receipt. Refunds are processed within 5 to 7 business days after we "
         "receive the returned item. For damaged or defective products, we offer an immediate "
         "replacement or a full refund at no extra cost. Partial refunds may be granted for "
         "items showing signs of use. Sale items and digital downloads are non-refundable. "
         "To initiate a return, log in to your account and go to My Orders, then select "
         "Return Item next to the relevant product."),

        ("2. Shipping & Delivery",
         "We offer three shipping tiers: Standard shipping takes 5 to 7 business days and is "
         "free for orders above Rs. 999. Express shipping delivers in 2 to 3 business days "
         "and costs Rs. 199. Overnight delivery is available for Rs. 499 and ships the next "
         "business day. Orders placed before 2 PM IST are dispatched the same day. Tracking "
         "information is sent via email and SMS within 24 hours of dispatch. Delivery to "
         "remote pin codes may take 1 to 2 additional days. International shipping is "
         "available to over 30 countries at rates calculated at checkout."),

        ("3. Account & Password Reset",
         "To reset your password, click the Forgot Password link on the login page and enter "
         "your registered email address. A secure password reset link will arrive within "
         "5 minutes. The link expires after 1 hour for security reasons. If the email does "
         "not arrive, check your spam or promotions folder. After 5 consecutive failed login "
         "attempts, your account will be temporarily locked. To unlock it, contact our "
         "support team with your registered email and a government-issued photo ID for "
         "identity verification. Two-factor authentication is available under Account "
         "Security settings and is strongly recommended for all users."),

        ("4. Payment Methods Accepted",
         "We accept all major credit and debit cards including Visa, MasterCard, American "
         "Express, and RuPay. UPI payments are supported through Google Pay, PhonePe, and "
         "Paytm. Net banking is available for all major Indian banks including SBI, HDFC, "
         "ICICI, and Axis Bank. Zero-cost EMI options are available for orders above Rs. 3000 "
         "with select partner banks for 3, 6, or 12-month tenures. Cash on Delivery is "
         "available for orders below Rs. 5000 in eligible pin codes. All transactions use "
         "256-bit SSL encryption. We do not store card numbers on our servers."),

        ("5. Order Cancellation Policy",
         "Orders can be cancelled within 2 hours of placement for a guaranteed full refund. "
         "After 2 hours, cancellation is only accepted if the order has not yet been "
         "dispatched from our warehouse. To cancel an order, go to My Orders in your account "
         "dashboard and click Cancel Order next to the product. Refunds for cancelled orders "
         "are credited within 3 to 5 business days to the original payment method. For "
         "prepaid orders cancelled after dispatch, you must refuse delivery; the refund is "
         "initiated once the package returns to our warehouse, typically within 7 to 10 days."),

        ("6. Product Warranty",
         "All electronics sold on our platform carry a minimum 1-year manufacturer warranty "
         "covering hardware defects and component failures under normal use conditions. "
         "Extended warranty plans of 2 or 3 years are available for purchase at checkout. "
         "Warranty coverage does not extend to physical damage, liquid damage, unauthorized "
         "repairs, or cosmetic wear and tear. To file a warranty claim, contact our support "
         "team with your order number, date of purchase, and a clear description of the "
         "issue. We may request photographs or a short video of the defect. Approved claims "
         "are resolved via repair, replacement, or refund within 7 business days."),

        ("7. Customer Support Contact Hours",
         "Our customer support team is available Monday to Saturday from 9 AM to 6 PM IST. "
         "You can reach us via email at support@shopify-demo.com with responses guaranteed "
         "within 24 hours. Live chat is available on our website homepage during business "
         "hours for instant assistance. Our toll-free phone support line 1800-000-0000 is "
         "available during business hours only. Outside business hours, our AI assistant "
         "handles common queries automatically. Gold tier loyalty members receive priority "
         "support with guaranteed 4-hour response times on all channels."),

        ("8. Loyalty Rewards Program",
         "Our loyalty program rewards customers with 1 point for every Rs. 100 spent on "
         "eligible purchases. Points are redeemable at the rate of Rs. 1 per point, with "
         "no minimum redemption amount. Membership tiers: Silver members with 1000 or more "
         "points receive a 5 percent extra discount on all orders. Gold members with 5000 or "
         "more points receive a 10 percent discount, free express shipping on all orders, and "
         "priority customer support access. Points expire after 12 months of account "
         "inactivity. Bonus points are awarded during seasonal sales events and for writing "
         "verified product reviews, at 50 points per approved review."),
    ]

    for title, text in sections:
        story.append(Paragraph(title, h1))
        story.append(Paragraph(text, body))
        story.append(Spacer(1, 0.15 * inch))

    doc.build(story)
    print(f"✅ Knowledge base PDF created: {path}")
    print(f"   Sections : {len(sections)}")
    return path


if __name__ == "__main__":
    create_knowledge_base()
