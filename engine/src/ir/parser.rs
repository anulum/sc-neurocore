//! Text-format parser for SC IR graphs.
//!
//! Parses the format produced by `printer::print()`. The parser is
//! intentionally simple (line-oriented) since the format is machine-
//! generated. A future version may support full MLIR-compatible syntax.

use crate::ir::graph::*;

/// Parse error with line number.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

/// Parse an SC IR text file into a graph.
///
/// This parser handles the subset of the text format needed for
/// round-trip testing: `sc.input`, `sc.output`, `sc.constant`, `sc.encode`,
/// `sc.and`, `sc.popcount`, and `sc.dense_forward`.
///
/// Complex ops (`LifStep`, `Scale`, `Offset`, `DivConst`) are parsed by
/// recognizing the op name and extracting key-value parameters.
pub fn parse(text: &str) -> Result<ScGraph, ParseError> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Err(ParseError {
            line: 0,
            message: "empty input".to_string(),
        });
    }

    // Line 0: "sc.graph @name {"
    let first = lines[0].trim();
    let name = first
        .strip_prefix("sc.graph @")
        .and_then(|s| s.strip_suffix(" {"))
        .ok_or_else(|| ParseError {
            line: 1,
            message: "expected 'sc.graph @name {'".to_string(),
        })?
        .to_string();

    let mut graph = ScGraph::new(name);

    for (line_idx, line) in lines.iter().enumerate().skip(1) {
        let trimmed = line.trim();
        if trimmed == "}" || trimmed.is_empty() {
            continue;
        }

        if trimmed.contains("= sc.input") {
            parse_input(trimmed, &mut graph, line_idx + 1)?;
        } else if trimmed.starts_with("sc.output") {
            parse_output(trimmed, &mut graph, line_idx + 1)?;
        } else if trimmed.contains("= sc.constant") {
            parse_constant(trimmed, &mut graph, line_idx + 1)?;
        } else if trimmed.contains("= sc.encode") {
            parse_encode(trimmed, &mut graph, line_idx + 1)?;
        } else if trimmed.contains("= sc.and") {
            parse_and(trimmed, &mut graph, line_idx + 1)?;
        } else if trimmed.contains("= sc.popcount") {
            parse_popcount(trimmed, &mut graph, line_idx + 1)?;
        } else if trimmed.contains("= sc.dense_forward") {
            parse_dense_forward(trimmed, &mut graph, line_idx + 1)?;
        } else if trimmed.contains("= sc.lif_step") {
            parse_lif_step(trimmed, &mut graph, line_idx + 1)?;
        } else if trimmed.contains("= sc.scale") {
            parse_scale(trimmed, &mut graph, line_idx + 1)?;
        } else if trimmed.contains("= sc.offset") {
            parse_offset(trimmed, &mut graph, line_idx + 1)?;
        } else if trimmed.contains("= sc.div_const") {
            parse_div_const(trimmed, &mut graph, line_idx + 1)?;
        } else {
            return Err(ParseError {
                line: line_idx + 1,
                message: format!("unrecognised op: {}", trimmed),
            });
        }
    }

    Ok(graph)
}

// Helpers

fn parse_value_id(s: &str) -> Result<ValueId, String> {
    let s = s.trim().trim_matches(',');
    s.strip_prefix('%')
        .and_then(|n| n.parse::<u32>().ok())
        .map(ValueId)
        .ok_or_else(|| format!("invalid ValueId: '{}'", s))
}

fn parse_type(s: &str) -> Result<ScType, String> {
    let s = s.trim();
    if s == "rate" {
        return Ok(ScType::Rate);
    }
    if s == "bool" {
        return Ok(ScType::Bool);
    }
    if s == "u64" {
        return Ok(ScType::UInt { width: 64 });
    }
    if let Some(w) = s.strip_prefix('u') {
        if let Ok(width) = w.parse::<u32>() {
            return Ok(ScType::UInt { width });
        }
    }
    if let Some(w) = s.strip_prefix('i') {
        if let Ok(width) = w.parse::<u32>() {
            return Ok(ScType::SInt { width });
        }
    }
    if let Some(inner) = s
        .strip_prefix("bitstream<")
        .and_then(|r| r.strip_suffix('>'))
    {
        let length = inner.parse::<usize>().map_err(|e| e.to_string())?;
        return Ok(ScType::Bitstream { length });
    }
    if s == "bitstream" {
        return Ok(ScType::Bitstream { length: 0 }); // unspecified
    }
    if let Some(inner) = s.strip_prefix("fixed<").and_then(|r| r.strip_suffix('>')) {
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() == 2 {
            let width = parts[0].trim().parse::<u32>().map_err(|e| e.to_string())?;
            let frac = parts[1].trim().parse::<u32>().map_err(|e| e.to_string())?;
            return Ok(ScType::FixedPoint { width, frac });
        }
    }
    if let Some(inner) = s.strip_prefix("vec<").and_then(|r| r.strip_suffix('>')) {
        // "bool,7" -> Vec<Bool, 7>
        if let Some(comma_pos) = inner.rfind(',') {
            let elem_str = &inner[..comma_pos];
            let count_str = inner[comma_pos + 1..].trim();
            let element = parse_type(elem_str)?;
            let count = count_str.parse::<usize>().map_err(|e| e.to_string())?;
            return Ok(ScType::Vec {
                element: Box::new(element),
                count,
            });
        }
    }
    Err(format!("unrecognised type: '{}'", s))
}

fn extract_kv(text: &str, key: &str) -> Option<String> {
    text.find(&format!("{}=", key)).map(|start| {
        let rest = &text[start + key.len() + 1..];
        let end = rest.find([',', ' ', ':']).unwrap_or(rest.len());
        rest[..end].to_string()
    })
}

fn make_err(line: usize, msg: impl Into<String>) -> ParseError {
    ParseError {
        line,
        message: msg.into(),
    }
}

fn parse_scalar_constant(val_str: &str, ty: &ScType, line: usize) -> Result<ScConst, ParseError> {
    if val_str.contains('.') || matches!(ty, ScType::Rate) {
        return val_str
            .parse::<f64>()
            .map(ScConst::F64)
            .map_err(|e| make_err(line, e.to_string()));
    }
    match ty {
        ScType::FixedPoint { .. } | ScType::SInt { .. } => val_str
            .parse::<i64>()
            .map(ScConst::I64)
            .map_err(|e| make_err(line, e.to_string())),
        _ => val_str
            .parse::<u64>()
            .map(ScConst::U64)
            .map_err(|e| make_err(line, e.to_string())),
    }
}

fn parse_vector_constant(val_str: &str, line: usize) -> Result<ScConst, ParseError> {
    let inner = val_str
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| make_err(line, "malformed vector constant"))?;
    if inner.trim().is_empty() {
        return Ok(ScConst::I64Vec(Vec::new()));
    }
    let is_float = inner.split(',').any(|part| part.trim().contains('.'));
    if is_float {
        let mut out = Vec::new();
        for token in inner.split(',') {
            out.push(
                token
                    .trim()
                    .parse::<f64>()
                    .map_err(|e| make_err(line, e.to_string()))?,
            );
        }
        Ok(ScConst::F64Vec(out))
    } else {
        let mut out = Vec::new();
        for token in inner.split(',') {
            out.push(
                token
                    .trim()
                    .parse::<i64>()
                    .map_err(|e| make_err(line, e.to_string()))?,
            );
        }
        Ok(ScConst::I64Vec(out))
    }
}

// Op parsers

fn parse_input(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    // %0 = sc.input "x_in" : rate
    let parts: Vec<&str> = text.splitn(2, "= sc.input").collect();
    if parts.len() != 2 {
        return Err(make_err(line, "malformed sc.input"));
    }
    let id = parse_value_id(parts[0]).map_err(|e| make_err(line, e))?;
    let rest = parts[1].trim();

    // Extract name between quotes
    let name_start = rest
        .find('"')
        .ok_or_else(|| make_err(line, "missing name"))?;
    let name_end = rest[name_start + 1..]
        .find('"')
        .ok_or_else(|| make_err(line, "unterminated name"))?;
    let name = rest[name_start + 1..name_start + 1 + name_end].to_string();

    // Extract type after ':'
    let colon_pos = rest
        .rfind(':')
        .ok_or_else(|| make_err(line, "missing type"))?;
    let ty = parse_type(&rest[colon_pos + 1..]).map_err(|e| make_err(line, e))?;

    graph.next_id = graph.next_id.max(id.0 + 1);
    graph.push(ScOp::Input { id, name, ty });
    Ok(())
}

fn parse_output(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    // sc.output "result" %5
    let rest = text.strip_prefix("sc.output").unwrap_or(text).trim();
    let name_start = rest
        .find('"')
        .ok_or_else(|| make_err(line, "missing name"))?;
    let name_end = rest[name_start + 1..]
        .find('"')
        .ok_or_else(|| make_err(line, "unterminated name"))?;
    let name = rest[name_start + 1..name_start + 1 + name_end].to_string();

    let after_name = rest[name_start + 1 + name_end + 1..].trim();
    let source = parse_value_id(after_name).map_err(|e| make_err(line, e))?;

    let id = ValueId(graph.next_id);
    graph.next_id += 1;
    graph.push(ScOp::Output { id, name, source });
    Ok(())
}

fn parse_constant(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    let parts: Vec<&str> = text.splitn(2, "= sc.constant").collect();
    if parts.len() != 2 {
        return Err(make_err(line, "malformed sc.constant"));
    }
    let id = parse_value_id(parts[0]).map_err(|e| make_err(line, e))?;
    let rest = parts[1].trim();

    let colon_pos = rest
        .rfind(':')
        .ok_or_else(|| make_err(line, "missing type"))?;
    let val_str = rest[..colon_pos].trim();
    let ty = parse_type(&rest[colon_pos + 1..]).map_err(|e| make_err(line, e))?;

    let value = if val_str.starts_with('[') {
        parse_vector_constant(val_str, line)?
    } else {
        parse_scalar_constant(val_str, &ty, line)?
    };

    graph.next_id = graph.next_id.max(id.0 + 1);
    graph.push(ScOp::Constant { id, value, ty });
    Ok(())
}

fn parse_encode(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    let parts: Vec<&str> = text.splitn(2, "= sc.encode").collect();
    if parts.len() != 2 {
        return Err(make_err(line, "malformed sc.encode"));
    }
    let id = parse_value_id(parts[0]).map_err(|e| make_err(line, e))?;
    let rest = parts[1].trim();

    // First token after "= sc.encode " is the prob operand.
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    let prob = parse_value_id(
        tokens
            .first()
            .ok_or_else(|| make_err(line, "missing prob"))?,
    )
    .map_err(|e| make_err(line, e))?;

    let length_str = extract_kv(rest, "length").ok_or_else(|| make_err(line, "missing length"))?;
    let length = length_str
        .parse::<usize>()
        .map_err(|e| make_err(line, e.to_string()))?;

    let seed_str = extract_kv(rest, "seed").ok_or_else(|| make_err(line, "missing seed"))?;
    let seed = if seed_str.starts_with("0x") || seed_str.starts_with("0X") {
        u16::from_str_radix(&seed_str[2..], 16).map_err(|e| make_err(line, e.to_string()))?
    } else {
        seed_str
            .parse::<u16>()
            .map_err(|e| make_err(line, e.to_string()))?
    };

    graph.next_id = graph.next_id.max(id.0 + 1);
    graph.push(ScOp::Encode {
        id,
        prob,
        length,
        seed,
    });
    Ok(())
}

fn parse_and(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    let parts: Vec<&str> = text.splitn(2, "= sc.and").collect();
    if parts.len() != 2 {
        return Err(make_err(line, "malformed sc.and"));
    }
    let id = parse_value_id(parts[0]).map_err(|e| make_err(line, e))?;
    let rest = parts[1].trim();
    let operands: Vec<&str> = rest.split(':').next().unwrap_or("").split(',').collect();
    if operands.len() < 2 {
        return Err(make_err(line, "sc.and needs 2 operands"));
    }
    let lhs = parse_value_id(operands[0]).map_err(|e| make_err(line, e))?;
    let rhs = parse_value_id(operands[1]).map_err(|e| make_err(line, e))?;

    graph.next_id = graph.next_id.max(id.0 + 1);
    graph.push(ScOp::BitwiseAnd { id, lhs, rhs });
    Ok(())
}

fn parse_popcount(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    let parts: Vec<&str> = text.splitn(2, "= sc.popcount").collect();
    if parts.len() != 2 {
        return Err(make_err(line, "malformed sc.popcount"));
    }
    let id = parse_value_id(parts[0]).map_err(|e| make_err(line, e))?;
    let rest = parts[1].trim();
    let input_str = rest.split(':').next().unwrap_or("").trim();
    let input = parse_value_id(input_str).map_err(|e| make_err(line, e))?;

    graph.next_id = graph.next_id.max(id.0 + 1);
    graph.push(ScOp::Popcount { id, input });
    Ok(())
}

fn parse_dense_forward(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    let parts: Vec<&str> = text.splitn(2, "= sc.dense_forward").collect();
    if parts.len() != 2 {
        return Err(make_err(line, "malformed sc.dense_forward"));
    }
    let id = parse_value_id(parts[0]).map_err(|e| make_err(line, e))?;
    let rest = parts[1].trim();

    let tokens: Vec<&str> = rest.split_whitespace().collect();
    let inputs = parse_value_id(
        tokens
            .first()
            .ok_or_else(|| make_err(line, "missing inputs"))?,
    )
    .map_err(|e| make_err(line, e))?;

    let weights_str =
        extract_kv(rest, "weights").ok_or_else(|| make_err(line, "missing weights"))?;
    let weights = parse_value_id(&weights_str).map_err(|e| make_err(line, e))?;

    let leak_str = extract_kv(rest, "leak").ok_or_else(|| make_err(line, "missing leak"))?;
    let leak = parse_value_id(&leak_str).map_err(|e| make_err(line, e))?;

    let gain_str = extract_kv(rest, "gain").ok_or_else(|| make_err(line, "missing gain"))?;
    let gain = parse_value_id(&gain_str).map_err(|e| make_err(line, e))?;

    let ni = extract_kv(rest, "ni")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    let nn = extract_kv(rest, "nn")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(7);
    let len = extract_kv(rest, "len")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1024);

    let params = DenseParams {
        n_inputs: ni,
        n_neurons: nn,
        stream_length: len,
        ..DenseParams::default()
    };

    graph.next_id = graph.next_id.max(id.0 + 1);
    graph.push(ScOp::DenseForward {
        id,
        inputs,
        weights,
        leak,
        gain,
        params,
    });
    Ok(())
}

fn parse_lif_step(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    let parts: Vec<&str> = text.splitn(2, "= sc.lif_step").collect();
    if parts.len() != 2 {
        return Err(make_err(line, "malformed sc.lif_step"));
    }
    let id = parse_value_id(parts[0]).map_err(|e| make_err(line, e))?;
    let rest = parts[1].trim();

    let tokens: Vec<&str> = rest.split_whitespace().collect();
    let current = parse_value_id(
        tokens
            .first()
            .ok_or_else(|| make_err(line, "missing current"))?,
    )
    .map_err(|e| make_err(line, e))?;

    let leak_str = extract_kv(rest, "leak").ok_or_else(|| make_err(line, "missing leak"))?;
    let leak = parse_value_id(&leak_str).map_err(|e| make_err(line, e))?;

    let gain_str = extract_kv(rest, "gain").ok_or_else(|| make_err(line, "missing gain"))?;
    let gain = parse_value_id(&gain_str).map_err(|e| make_err(line, e))?;

    let noise_str = extract_kv(rest, "noise").ok_or_else(|| make_err(line, "missing noise"))?;
    let noise = parse_value_id(&noise_str).map_err(|e| make_err(line, e))?;

    let dw = extract_kv(rest, "dw")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(16);
    let frac = extract_kv(rest, "frac")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(8);
    let vt = extract_kv(rest, "vt")
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(256);
    let rp = extract_kv(rest, "rp")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(2);

    let params = LifParams {
        data_width: dw,
        fraction: frac,
        v_threshold: vt,
        refractory_period: rp,
        ..LifParams::default()
    };

    graph.next_id = graph.next_id.max(id.0 + 1);
    graph.push(ScOp::LifStep {
        id,
        current,
        leak,
        gain,
        noise,
        params,
    });
    Ok(())
}

fn parse_scale(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    let parts: Vec<&str> = text.splitn(2, "= sc.scale").collect();
    if parts.len() != 2 {
        return Err(make_err(line, "malformed sc.scale"));
    }
    let id = parse_value_id(parts[0]).map_err(|e| make_err(line, e))?;
    let rest = parts[1].trim();

    let tokens: Vec<&str> = rest.split_whitespace().collect();
    let input = parse_value_id(
        tokens
            .first()
            .ok_or_else(|| make_err(line, "missing input"))?,
    )
    .map_err(|e| make_err(line, e))?;

    let factor_str = extract_kv(rest, "factor").ok_or_else(|| make_err(line, "missing factor"))?;
    let factor = factor_str
        .parse::<f64>()
        .map_err(|e| make_err(line, e.to_string()))?;

    graph.next_id = graph.next_id.max(id.0 + 1);
    graph.push(ScOp::Scale { id, input, factor });
    Ok(())
}

fn parse_offset(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    let parts: Vec<&str> = text.splitn(2, "= sc.offset").collect();
    if parts.len() != 2 {
        return Err(make_err(line, "malformed sc.offset"));
    }
    let id = parse_value_id(parts[0]).map_err(|e| make_err(line, e))?;
    let rest = parts[1].trim();

    let tokens: Vec<&str> = rest.split_whitespace().collect();
    let input = parse_value_id(
        tokens
            .first()
            .ok_or_else(|| make_err(line, "missing input"))?,
    )
    .map_err(|e| make_err(line, e))?;

    let offset_str = extract_kv(rest, "offset").ok_or_else(|| make_err(line, "missing offset"))?;
    let offset = offset_str
        .parse::<f64>()
        .map_err(|e| make_err(line, e.to_string()))?;

    graph.next_id = graph.next_id.max(id.0 + 1);
    graph.push(ScOp::Offset { id, input, offset });
    Ok(())
}

fn parse_div_const(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    let parts: Vec<&str> = text.splitn(2, "= sc.div_const").collect();
    if parts.len() != 2 {
        return Err(make_err(line, "malformed sc.div_const"));
    }
    let id = parse_value_id(parts[0]).map_err(|e| make_err(line, e))?;
    let rest = parts[1].trim();

    let tokens: Vec<&str> = rest.split_whitespace().collect();
    let input = parse_value_id(
        tokens
            .first()
            .ok_or_else(|| make_err(line, "missing input"))?,
    )
    .map_err(|e| make_err(line, e))?;

    let divisor_str =
        extract_kv(rest, "divisor").ok_or_else(|| make_err(line, "missing divisor"))?;
    let divisor = divisor_str
        .parse::<u64>()
        .map_err(|e| make_err(line, e.to_string()))?;

    graph.next_id = graph.next_id.max(id.0 + 1);
    graph.push(ScOp::DivConst { id, input, divisor });
    Ok(())
}
