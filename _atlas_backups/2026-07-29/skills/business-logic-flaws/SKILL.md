---
name: business-logic-flaws
description: "Business logic vulnerability testing — coupon abuse, price manipulation, race conditions, workflow bypass. Scanner-invisible, high-reward."
version: 1.0.0
---

# Business Logic Flaw Testing

## Categories

### 1. Coupon / Discount Abuse
- Apply expired coupons
- Stack multiple coupons (apply coupon1, then coupon2)
- Apply coupon to ineligible items
- Reuse single-use coupon codes
- Race condition: apply same coupon twice simultaneously
- Negative discounts: `{"discount": -100}`

### 2. Price Manipulation
- Modify price field in POST/PUT requests
- Negative prices: `{"price": -100}`
- Fractional amounts: `{"quantity": 0.001}`
- Modify currency
- Modify shipping cost

### 3. Quantity Manipulation
- Negative quantities: `{"quantity": -1}` (might give refund/credit)
- Zero quantities: `{"quantity": 0}`
- Overflow: `{"quantity": 999999999}`
- Fractional: `{"quantity": 0.5}`

### 4. Race Conditions
- Apply same coupon twice in parallel
- Spend same gift card twice
- Register same promo code twice
- Withdraw money twice before balance updates

### 5. Workflow Bypass
- Skip payment step
- Go directly to order confirmation
- Access admin functions without authentication
- Complete multi-step flow out of order

### 6. Trial/Rate Limit Abuse
- Reset trial period by changing account attributes
- Use different payment methods to restart trials
- Brute force with rotating proxies/IPs

### 7. State Machine Violations
- Cancel an already-cancelled order
- Refund an unshipped item
- Approve a rejected request
- Move backwards in a workflow
