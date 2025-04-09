# sky-accessibility-rewards

# Referral Net Tracker Algorithm Documentation

## Overview

The Referral Net Tracker algorithm tracks funds associated with integrator referral codes and calculates time-weighted rewards based on fund retention periods. This document explains the core concepts, workflow, and calculation methodology used by the algorithm.

## Key Concepts

### Pots
A "pot" is the fundamental tracking unit that represents a discrete amount of funds with the following attributes:
- **Amount**: The quantity of tokens
- **Referral Code**: The integrator code that brought these funds into the system
- **Address**: The wallet address that owns these funds
- **Contract**: The smart contract these funds are associated with
- **Liquidity Token**: The token type in this pot (currently only sUSDS, would allow for cross-farm same LP/stake token issueing to be respected)

### Events
The algorithm processes several types of blockchain events:
- **Referral/Stake/Deposit**: Events that create new pots
- **Withdraw**: Events that reduce pot amounts
- **Transfer**: Events that move funds between addresses while preserving referral attribution

### Time-Weighted Value
Rather than measuring static balances, the algorithm calculates value based on both amount and duration, expressed as "token-days" or equivalent.

## Algorithm Workflow

1. **Initialization**
   - Create data structures for tracking pots and their history
   - Initialize monthly tracking for time-series analysis

2. **Chronological Event Processing**
   - Process all blockchain events in strict timestamp order
   - At month boundaries, create checkpoints for consistent time-series analysis

3. **Event Handling**
   - For referral/stake events: Create new pots with appropriate attribution
   - For withdrawal events: Reduce pot amounts, maintaining referral attribution
   - For transfer events: Withdraw from source address and create new pot for destination

4. **Monthly Snapshots**
   - Record complete state of all pots at month boundaries
   - Create empty checkpoints for all referral code and contract combinations

5. **Reward Calculation**
   - Calculate time-weighted balances for each referral code
   - Apply reward percentage to determine payout amounts

## TVL Calculation Methodology

The TVL (Total Value Locked) per integrator code is calculated using time-weighted methodology:

1. **Balance History Collection**
   - Record all balance-changing events with precise timestamps
   - Apply referral code attribution to each event

2. **Time-Weighted Calculation**
   - For each referral code and contract combination:
     - Calculate running balance at each time point
     - Multiply each balance by the time period it was maintained
     - Express time periods as fraction of a year for annualization

3. **Aggregation and Conversion**
   - Group time-weighted values by month and referral code
   - For sUSDS tokens, apply price conversion to normalize to USD value
   - Sum across all contracts to get total TVL per referral code

4. **Reward Application**
   - Apply reward percentage to time-weighted TVL
   - Calculate final payout amount per referral code per month

## Technical Implementation Notes

1. **Referral Attribution Preservation**
   - When funds are withdrawn, the algorithm carefully preserves referral attribution
   - If a withdrawal spans multiple pots with different referral codes, attribution is proportionally maintained

2. **Fund Tracing**
   - The algorithm maintains transaction traces to ensure full auditability
   - Each pot keeps record of its originating transaction and related pots

3. **Data Integrity**
   - Sanity checks verify that calculated balances match expected totals
   - Snapshots at month boundaries ensure consistent time-series analysis

4. **Edge Case Handling**
   - Small discrepancies (<1 $ worht of token) are logged but not treated as errors
   - Larger discrepancies trigger alerts for manual investigation
§
## Example Calculation

For a given referral code in a single month:
1. Sum all deposits with this code: 1000 tokens
2. If these tokens remain for the entire month (30 days): 
   - Time-weighted value = 1000 × (30/365) = 82.19 token-years
3. Apply reward percentage (e.g., 0.4%):
   - Monthly reward = 82.19 × 0.4% = 0.33 tokens

This methodology ensures rewards are proportional to both the amount of funds and their retention duration, creating aligned incentives for sustainable growth.

# Detailed Explanation of Pot Finding and Division in the Referral Net Tracker Algorithm

The pot system is the core of the Referral Net Tracker algorithm's attribution mechanism. Here's a detailed explanation of how pots are found and divided when funds move in the system:

## Finding Pots

The algorithm uses the `find_pot` function to locate appropriate pots when funds need to be withdrawn:

```python
def find_pot(from_address, ps, code=None, liquidity_token=None, contract=None):
    for p in ps:
        if (
            p['address'] == from_address 
            and (code == None or p['referral_code'] == code) 
            and p['amount'] > 0 
            and (liquidity_token == p['liquidity_token'] or liquidity_token == None)
            and (contract == None or contract == p['contract'])
        ):
            return p
    return None
```

This function searches through all existing pots with multiple criteria:

1. **Address matching**: The pot must belong to the address that's withdrawing funds
2. **Referral code matching** (optional): If specified, only pots with this referral code are considered
3. **Positive balance**: The pot must have funds available (amount > 0)
4. **Token type matching** (optional): If specified, only pots with this liquidity token are considered
5. **Contract matching** (optional): If specified, only pots from this contract are considered

## Dividing Pots During Withdrawals

The division of pots during withdrawals is handled in the `withdrawFromPot` function, which implements a sophisticated attribution-preserving algorithm:

### The Withdrawal Process

1. **Initial Pot Search**:
   ```python
   f_pot = find_pot(pot_og_address, pots, contract=p_contract, liquidity_token=p_liquidity_token)
   ```
   The algorithm first attempts to find a pot belonging to the withdrawing address with matching criteria.

2. **First Pot Reduction**:
   ```python
   reduce = min(left, f_pot['amount'])
   left -= reduce
   transferred += reduce
   f_pot['amount'] -= reduce
   ```
   It reduces this pot by the amount being withdrawn, or the entire pot if the withdrawal is larger.

3. **Multi-Pot Processing**:
   If the withdrawal amount exceeds the first pot's balance, the algorithm enters a loop to find additional pots:
   ```python
   while left > 0 and f_pot:
       f_pot = find_pot(pot_og_address, pots, contract=p_contract, liquidity_token=p_liquidity_token)
       # Process this pot
   ```

4. **Referral Code Transition**:
   ```python
   if f_pot['referral_code'] != refcode:
       # Handle referral code change
   ```
   This is where the algorithm carefully manages attribution when withdrawing from pots with different referral codes.

### Handling Different Referral Codes

When the algorithm encounters pots with different referral codes during withdrawal:

1. **For Transfers**:
   - It creates a new pot for the destination address using the first referral code
   - Records the transfer with proper attribution
   - Resets the accumulated amount and starts tracking the new referral code
   
   ```python
   if change_type == 'transfer':
       p = create_new_pot(refcode, parsedEvent.to_address, transferred, 
                        parsedEvent.tx_hash, contract, liquidity_token, spawn='transfer')
       # Record transfer and reset for next referral code
   ```

2. **For Withdrawals**:
   - Records a balance change for the current referral code's portion
   - Resets the accumulated amount
   - Continues with the new referral code
   
   ```python
   else:  # withdrawal
       balance_change = {'ref_code': str(refcode), 'amount': transferred * -1, ...}
       balance_changes.append(balance_change)
       transferred = 0
       refcode = f_pot['referral_code']
   ```

### Example of Pot Division

Let's walk through a concrete example:

1. User has three pots in the following order:
   - Pot A: 500 tokens with referral code 1001
   - Pot B: 300 tokens with referral code 1002
   - Pot C: 200 tokens with referral code 1001

2. User withdraws 600 tokens:
   - Algorithm first reduces Pot A: -500 tokens (fully depleted)
   - Records: 500 tokens withdrawn from code 1001
   - Still needs 100 more tokens
   - Finds Pot B with different code (1002)
   - Records: 100 tokens withdrawn from code 1002
   - Pot B now has 200 tokens remaining
   - Withdrawal complete

This ensures each referral code gets proper attribution for the exact amount of funds they brought in, even during partial withdrawals. The algorithm maintains an accounting system where proportional attribution is preserved regardless of how users move their funds.

## Special Cases in Pot Division

1. **Insufficient Funds**:
   If the algorithm can't find enough pots to cover the full withdrawal:
   ```python
   if left > 0:
       print("ERROR left over in withdraw", currentTxHash, left)
   ```
   It logs an error, which may indicate unexpected blockchain state or a potential issue requiring investigation.

2. **Small Discrepancies**:
   For very small discrepancies (< 1 $ token), the algorithm takes a pragmatic approach:
   ```python
   if left/1e18 > 1:
       end = True  # Major error, should stop processing
   else:
       print('ERROR too small, ignored')  # Minor error, can continue
   ```
   This prevents the entire process from failing due to minor rounding or dust amounts.

This pot finding and division mechanism enables precise attribution tracking throughout the life cycle of funds in the system, forming the foundation for accurate time-weighted reward calculations.

# Relationship Between Pots and Rewards Calculation

The Referral Net Tracker algorithm establishes a direct relationship between the pot tracking system and the reward calculation methodology. Here's a detailed explanation of how pots are used to calculate rewards:

## 1. From Pots to Balance History

While pots track the real-time state of funds, the reward calculation requires a historical record of all balance changes:

```python
# Each pot operation (creation, reduction) generates entries in the balance history
balance_change = {
    'ref_code': parsedEvent.code, 
    'amount': parsedEvent.value, 
    'timestamp': parsedEvent.row['block_timestamp'],
    'block_number': parsedEvent.row['block_number'],
    'event_index': parsedEvent.event_index,
    'to': parsedEvent.from_address,
    'type': 'referral',
    'contract': parsedEvent.contract, 
    'booking_num': 0,
    'tx': currentTxHash
}
balance_history.append(balance_change)
```

This balance history creates a complete audit trail of how funds moved between pots over time, preserving crucial time and attribution data.

## 2. Time-Weighted Calculation Process

The reward calculation uses this balance history in the `calculate_rewards` function:

```python
def calculate_rewards(balance_history, dfsUSDSPrices, reward_percentage):
    # Convert to DataFrame
    dfBH = pd.DataFrame(balance_history)
    dfBH['ref_code'] = dfBH['ref_code'].astype(str)
    
    # Get unique code-contract combinations
    code_contract_uniques = dfBH[['contract', 'ref_code']].value_counts().reset_index()
    
    # Set annualization factor
    total_seconds = 60 * 60 * 24 * 365
```

The algorithm:

1. **Processes each referral code and contract combination separately:**
   ```python
   for i, r in enumerate(code_contract_uniques.iterrows()):
       current_contract = r[1]['contract']
       current_ref_code = r[1]['ref_code']
       
       # Filter for this code/contract
       dfBHt = dfBH[(dfBH['contract'] == current_contract) & 
                    (dfBH['ref_code'] == current_ref_code)].copy()
   ```

2. **Calculates the cumulative running balance over time:**
   ```python
   dfBHt.loc[:, 'cumsum'] = dfBHt['amount'].cumsum()
   ```
   This shows how much was in pots with this referral code at each point in time.

3. **Calculates the time periods between balance changes:**
   ```python
   dfBHt.loc[:, 'held_time'] = dfBHt['timestamp'].diff().replace(
       pd.NaT, pd.Timedelta(seconds=1)).dt.total_seconds() / total_seconds
   ```
   This converts time differences to fractions of a year (for annualization).

4. **Calculates the time-weighted values:**
   ```python
   dfBHt.loc[:, 'eligible_tvl'] = dfBHt['cumsum'] * dfBHt['held_time']
   ```
   This is the key calculation - multiplying the amount by how long it was held.

## 3. From Time-Weighted Values to Rewards

The actual reward calculation follows these steps:

1. **Group by month:**
   ```python
   dfReward = (dfBHt.groupby('month')['eligible_tvl'].sum() / 1e18).reset_index()
   ```
   This aggregates time-weighted values by month (converting from wei to tokens).

2. **Apply price conversion for sUSDS:**
   ```python
   if current_contract == '0xa3931d71877c0e7a3148cb7eb4463524fec27fbd':
       dfBHt.loc[:, 'eligible_tvl'] = dfBHt.apply(lambda x: rowToUSDValue(x), axis=1)
   ```
   This ensures sUSDS tokens are properly valued using price data.

3. **Calculate rewards:**
   ```python
   dfRAggPayout = dfRAgg.copy()
   dfRAggPayout['payout'] = dfRAggPayout['eligible_tvl'] * reward_percentage
   ```
   This applies the reward percentage (e.g., 0.4%) to the eligible TVL.

## 4. Concrete Example with Pots

Let's walk through a concrete example to illustrate:

1. **Initial State:**
   - User deposits 1000 tokens through referral code 1001
   - Algorithm creates a pot: `{potnum: 0, referral_code: 1001, amount: 1000, ...}`
   - Balance history records: `{ref_code: 1001, amount: 1000, timestamp: Jan 1, ...}`

2. **One Month Later:**
   - Balance is still 1000 tokens
   - Time elapsed: 30 days = 30/365 years ≈ 0.082 years
   - Time-weighted value: 1000 × 0.082 = 82 token-years

3. **Partial Withdrawal:**
   - On Jan 31, user withdraws 400 tokens
   - Pot is reduced: `{potnum: 0, referral_code: 1001, amount: 600, ...}`
   - Balance history records: `{ref_code: 1001, amount: -400, timestamp: Jan 31, ...}`
   - January cumulative calculation:
     - 1000 tokens for 30 days = 82 token-years
     - Final eligible TVL for January: 82 token-years

4. **Second Month:**
   - Starting balance: 600 tokens
   - Held for entire February (28 days): 28/365 years ≈ 0.077 years
   - Time-weighted value: 600 × 0.077 = 46.2 token-years
   - February eligible TVL: 46.2 token-years

5. **Final Reward Calculation:**
   - January reward: 82 token-years × 0.4% = 0.328 tokens
   - February reward: 46.2 token-years × 0.4% = 0.185 tokens

## 5. Key Insights on Pot-Reward Relationship

1. **Pots track instantaneous state, rewards track time-integrated value:**
   - Pots: "User X has Y tokens from referral Z right now"
   - Rewards: "Referral Z has accumulated X token-years of value this month"

2. **Reward attribution is preserved during pot divisions:**
   - When funds are withdrawn from multiple pots with different referral codes, each code's contribution to the withdrawal is separately tracked
   - This ensures rewards are fairly attributed even during complex withdrawal patterns

3. **Monthly boundaries are critical:**
   - Months are the reward calculation period
   - Pot snapshots at month boundaries ensure consistent reward calculations
   - Checkpoints at month boundaries enable proper time-weighted calculations

This methodology ensures that rewards accurately reflect both the amount of funds attributable to each referral code and how long those funds remained in the system, creating proper incentive alignment for sustainable growth.
