//! A small width-aware text table for CLI listings (`--list-engines`). One
//! column (e.g. a description) wraps to fit the terminal so long text doesn't
//! run past the right edge; the remaining columns are sized to their content and
//! the wrapped column's continuation lines leave them blank.

use terminal_size::{terminal_size, Width};

/// Terminal width in columns, or 100 when stdout is not a tty (piped/redirected),
/// floored so a very narrow terminal still leaves room for the wrap column.
fn term_cols() -> usize {
    terminal_size()
        .map(|(Width(w), _)| w as usize)
        .unwrap_or(100)
        .max(40)
}

/// Greedy word-wrap `text` to `width` columns. A word longer than `width` gets
/// its own (over-long) line rather than being broken mid-word.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if cur.is_empty() {
            cur.push_str(word);
        } else if cur.chars().count() + 1 + word.chars().count() <= width {
            cur.push(' ');
            cur.push_str(word);
        } else {
            lines.push(std::mem::take(&mut cur));
            cur.push_str(word);
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn pad(s: &str, w: usize) -> String {
    format!("{s}{}", " ".repeat(w.saturating_sub(s.chars().count())))
}

/// Render `rows` under `headers` as an aligned table. Column `wrap_col` wraps to
/// whatever width is left after the content-sized columns, using the terminal
/// width. Every row must have `headers.len()` cells.
pub fn render(headers: &[&str], rows: &[Vec<String>], wrap_col: usize) -> String {
    render_to(headers, rows, wrap_col, term_cols())
}

/// `render` with an explicit total width (for tests / fixed-width output).
fn render_to(headers: &[&str], rows: &[Vec<String>], wrap_col: usize, cols: usize) -> String {
    const GAP: usize = 2;
    let ncol = headers.len();

    // Content width of every column except the one that wraps.
    let mut width: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i != wrap_col {
                width[i] = width[i].max(cell.chars().count());
            }
        }
    }

    // The wrap column gets the leftover horizontal space (never below a floor).
    let fixed: usize = (0..ncol)
        .filter(|&i| i != wrap_col)
        .map(|i| width[i])
        .sum::<usize>()
        + GAP * (ncol - 1);
    let natural = rows
        .iter()
        .map(|r| r[wrap_col].chars().count())
        .max()
        .unwrap_or(0)
        .max(headers[wrap_col].chars().count());
    width[wrap_col] = natural.min(cols.saturating_sub(fixed)).max(20);

    let gap = " ".repeat(GAP);
    let mut out = String::new();

    let header: Vec<String> = (0..ncol).map(|i| pad(headers[i], width[i])).collect();
    out.push_str(header.join(&gap).trim_end());
    out.push('\n');
    let rule: Vec<String> = (0..ncol).map(|i| "─".repeat(width[i])).collect();
    out.push_str(&rule.join(&gap));
    out.push('\n');

    // Indent under the wrap column, for its continuation lines.
    let indent = " ".repeat((0..wrap_col).map(|i| width[i]).sum::<usize>() + GAP * wrap_col);

    for row in rows {
        let wrapped = wrap(&row[wrap_col], width[wrap_col]);
        for (li, wline) in wrapped.iter().enumerate() {
            if li == 0 {
                let cells: Vec<String> = (0..ncol)
                    .map(|i| pad(if i == wrap_col { wline } else { &row[i] }, width[i]))
                    .collect();
                out.push_str(cells.join(&gap).trim_end());
            } else {
                out.push_str(format!("{indent}{wline}").trim_end());
            }
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_description_to_width() {
        let rows = vec![vec![
            "strong".to_string(),
            "iterative-deepening PVS with aspiration windows and hash-move ordering".to_string(),
            "root-parallel scores".to_string(),
            "9".to_string(),
        ]];
        let out = render_to(
            &["name", "description", "parallelism", "depth"],
            &rows,
            1,
            60,
        );
        // No rendered line exceeds the requested width.
        for line in out.lines() {
            assert!(line.chars().count() <= 60, "line too wide: {line:?}");
        }
        // The description actually wrapped onto more than one line.
        assert!(out.lines().count() > 3, "expected wrapping:\n{out}");
        // Header, rule, and the fixed columns are present on the first row line.
        assert!(out.contains("name"));
        assert!(out.contains("root-parallel scores"));
        assert!(out.contains('─'));
    }

    #[test]
    fn fixed_columns_align_and_no_wrap_when_wide() {
        let rows = vec![
            vec![
                "a".to_string(),
                "short".to_string(),
                "seq".to_string(),
                "4".to_string(),
            ],
            vec![
                "bb".to_string(),
                "also short".to_string(),
                "seq".to_string(),
                "6".to_string(),
            ],
        ];
        let out = render_to(&["name", "desc", "par", "depth"], &rows, 1, 120);
        // Wide terminal => one line per row (header + rule + 2 rows = 4 lines).
        assert_eq!(out.lines().count(), 4, "no wrapping expected:\n{out}");
    }
}
