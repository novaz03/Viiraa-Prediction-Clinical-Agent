# Viiraa Blood Glucose Analysis — Frontend UI Design Improvement Plan

## Overview

This document proposes a frontend redesign direction for the **Viiraa Blood Glucose Analysis** interface. The current product already has the core functional pieces in place:

- user/session controls
- meal, lab, and demographic inputs
- pre-meal glucose series entry
- prediction outputs
- histogram comparisons
- verbal analysis
- research-use disclaimer

The main opportunity is not feature completeness, but **information architecture, hierarchy, and product polish**.

---

## Current UI Assessment

### What works
- The workflow contains all required inputs for prediction.
- Prediction results are clearly present.
- The interface includes both numeric outputs and narrative interpretation.
- Research-use-only language is visible.

### Main design problems
1. **Weak visual hierarchy**  
   Inputs, outputs, analysis, and charts all have similar visual weight.

2. **Overly long single-column feel**  
   The interface reads like a long internal tool form rather than a polished application.

3. **Results are not visually dominant**  
   The app's most valuable output is the prediction, but the layout does not strongly emphasize it.

4. **Chart areas feel empty or unfinished**  
   The histogram sections visually resemble placeholders.

5. **Technical actions compete with primary actions**  
   Buttons like JSON copy/download compete with the main prediction workflow.

---

## Primary Design Goal

Reframe the interface around a simple product story:

1. **Enter meal and glucose data**
2. **Review prediction and interpretation**

The user should immediately understand:
- what to enter
- what the app predicts
- how to interpret the result
- what level of confidence to place in it

---

## Recommended Layout

## 1. Use a two-column desktop layout

### Left column
Input workflow:
- Meal
- Demographics
- Labs
- Pre-meal glucose series

### Right column
Prediction workflow:
- Result summary
- Key metrics
- Comparison charts
- Interpretation
- Safety/disclaimer

### Why this helps
- Keeps the main outcome visible after submission
- Reduces scrolling
- Makes the experience feel analytical rather than form-heavy
- Better matches the mental model of “input on the left, result on the right”

---

## 2. Establish clear section order

Recommended page structure:

### Header
- App title
- Subtitle
- Research-use badge
- Optional user/session summary

### Main content
#### Inputs
- Meal information
- Demographics
- Lab values
- Pre-meal glucose data

#### Results
- Predicted excursion summary
- Metric cards
- Typical meal comparison
- Personal history comparison
- Verbal interpretation
- Safety note

---

## Input Form Redesign

## 3. Re-group the inputs into logical sections

### Meal
- Meal type
- Minutes since last meal
- Calories
- Carbs
- Protein
- Fat

### Demographics
- Age
- Gender
- Height
- Weight
- BMI (derived)

### Labs
- A1c
- Fasting glucose

### Pre-meal glucose series
- Large input area
- Format hint
- Validation state
- Preview sparkline

This structure is more scannable than mixing meal, labs, and demographics in a single broad block.

---

## 4. Improve the pre-meal glucose series experience

This is likely the most intimidating part of the UI.

### Recommended improvements
- Show a small sparkline preview after paste/input
- Show validation feedback such as:
  - number of points detected
  - valid/invalid format
  - missing values warning
- Provide accepted format examples
- Add a toggle for:
  - manual paste
  - CSV upload
- Place **Load Example** close to this section

### Why this matters
This field currently feels raw and technical. A guided input experience will reduce friction and improve trust.

---

## 5. Improve units and derived values

### Height and weight
Current presentation can be simplified.

Recommended:
- Height: feet/inches or cm toggle
- Weight: lb/kg toggle
- BMI: auto-calculated, visually secondary

### Design principle
Derived values should not compete visually with primary user-entered values.

---

## Results Redesign

## 6. Make the result summary the visual anchor

After prediction, the most prominent area should be a result summary card.

### Example structure
**Predicted Excursion: Moderate**

- Peak glucose: 143 mg/dL
- Peak increment: +31.9 mg/dL
- AUC 120: 14479
- iAUC 120: 1813
- 95% confidence range shown clearly

### Benefits
- Gives the user an immediate conclusion
- Creates a focal point
- Prevents the result from getting lost in the rest of the page

---

## 7. Redesign the metric cards

The three key outputs should be displayed as structured metric cards:

- Peak glucose
- AUC 120
- iAUC 120

### Each card should contain
- Large numeric value
- Unit or label
- Confidence interval in muted text
- Status badge
- Optional small trend or confidence bar

### Visual recommendations
- Use consistent spacing
- Keep labels short
- Use badge colors sparingly
- Avoid making the confidence interval compete with the main number

---

## 8. Standardize severity/status styling

Labels such as **LOW** and **MODERATE** should be visually consistent.

### Recommended system
- Neutral style for non-critical metadata
- Semantic badges for severity only
- One consistent badge pattern:
  - rounded pill
  - small uppercase or sentence case
  - consistent placement on all cards

### Suggested semantic palette
- Low: muted green
- Moderate: amber
- High: red

Do not overuse these colors elsewhere in the page.

---

## Chart and Comparison Redesign

## 9. Upgrade the histogram presentation

The comparison sections currently need much stronger chart framing.

### Problems to solve
- Too much empty gray space
- Weak visual payoff
- Limited interpretability at a glance

### Recommended improvements
- Add a visible histogram or density plot
- Show a vertical marker for the user's predicted value
- Include a one-line takeaway under each chart
- Add percentile callout in larger text
- Reduce unused container height
- Use stronger chart titles

Example:
- **Compared with Typical Lunch Meals**
- “Your predicted AUC is lower than about 76% of similar meals.”

---

## 10. Use tabs instead of long stacked comparisons

Instead of stacking many chart panels vertically, use:
- **Typical Meal Type**
- **Your Meal History**

This makes the result area shorter and easier to navigate.

Optional secondary tabs:
- AUC
- iAUC
- Peak amplitude

---

## Interpretation Layer

## 11. Rewrite the verbal analysis as a structured interpretation panel

The current narrative content is useful, but it should feel more like a high-value interpretation module.

### Recommended structure

#### Summary
Moderate post-meal excursion predicted.

#### What stands out
- Peak glucose is expected to rise noticeably
- Compared with similar meals, this is lower than typical
- Compared with personal history, this is somewhat lower

#### Uncertainty
Prediction intervals are broad, so interpret this as directional rather than exact.

#### Safety note
Research-use prediction only. Not diagnosis or treatment advice.

This structure is easier to scan than a plain paragraph block.

---

## Action and Control Hierarchy

## 12. Re-prioritize the buttons

Current actions likely include:
- Load Example
- Clear
- Run Prediction
- Copy Request JSON
- Download Result JSON

These should not all have similar visual weight.

### Recommended hierarchy
#### Primary action
- **Run Prediction**

#### Secondary action
- Load Example

#### Tertiary actions
- Clear
- Copy JSON
- Download JSON

### Better option
Move JSON tools into an **Advanced** disclosure panel.

That keeps the default UI focused on the user journey instead of developer tooling.

---

## Trust, Compliance, and Product Framing

## 13. Improve the research-use disclaimer presentation

This is a health-related interface. Compliance framing matters.

### Recommended approach
Use a dedicated info banner near the results header:

> Research-use only. These outputs are model predictions and not clinical diagnosis or treatment guidance.

### Why
- Makes the disclaimer visible without overwhelming the UI
- Improves trust
- Separates product interpretation from compliance language

---

## 14. Move technical metadata into an expandable debug area

Items like:
- request ID
- fixed backend parameters
- request JSON
- result JSON

should live inside:
- **Advanced**
- **Technical Details**
- **Model Request Details**

This is cleaner for most users while preserving audit/debug value.

---

## Visual Design System Recommendations

## 15. Improve typography and spacing

### Typography hierarchy
- Page title: large, strong
- Section title: medium, bold
- Card title: medium
- Field label: compact
- Helper text: smaller, muted

### Spacing
- Increase whitespace between major sections
- Reduce cramped spacing within cards
- Use consistent vertical rhythm

### Why this matters
Spacing is one of the fastest ways to make the interface feel more premium.

---

## 16. Use a more modern card system

### Recommended card treatment
- White surfaces
- Light border
- Subtle shadow
- Larger corner radius
- Cleaner separation between cards

### Result emphasis
The primary results card can have:
- slightly tinted background
- stronger elevation
- accent border or highlight

That gives the results the prominence they deserve.

---

## 17. Introduce a restrained accent color system

Use:
- one primary brand accent
- neutral grays for structure
- semantic colors only when conveying meaning

Avoid using multiple strong colors in parallel unless they encode true status.

---

## Suggested Target Experience

## 18. Proposed ideal screen flow

### Top
**Viiraa Blood Glucose Analysis**  
Research-use prediction of post-meal glucose excursion

### Left panel
**Enter Inputs**
- Meal
- Demographics
- Labs
- Pre-meal glucose data

### Right panel
**Prediction**
- Moderate excursion badge
- Key metrics
- Confidence intervals
- Typical meal comparison
- Personal history comparison
- Interpretation
- Safety note

### Bottom or expandable
**Advanced Details**
- request ID
- backend parameters
- JSON tools

---

## Priority Implementation Order

## 19. Best sequence for frontend improvement

### Phase 1 — highest impact
1. Two-column layout
2. Result summary redesign
3. Metric card redesign

### Phase 2
4. Chart container redesign
5. Verbal analysis redesign
6. Button hierarchy cleanup

### Phase 3
7. Pre-glucose input improvements
8. Advanced/debug panel separation
9. Polished typography, spacing, and theme system

---

## Final Recommendation

The app already has the right computational structure. The strongest frontend move is to shift from a **long analytical form** to an **outcome-centered product interface**.

### Core redesign principle
Make the prediction feel like the product, and make the form feel like a guided path to that prediction.

If implemented well, the UI will feel:
- more trustworthy
- easier to scan
- less internal-tool-like
- more appropriate for a research analytics application

---