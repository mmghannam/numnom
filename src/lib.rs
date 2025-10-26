pub use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::{self, Unexpected, Visitor};

/// Central MIP (Mixed Integer Programming) model structures
/// These can be used for any optimization format: MPS, LP, etc.
#[derive(Debug, Serialize, Deserialize)]
pub struct MipModel {
    pub name: String,
    pub variables: Vec<Variable>,
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub obj_coeff: f64,
    #[serde(rename = "type")]
    pub var_type: VariableType,
    #[serde(serialize_with = "serialize_bound", deserialize_with = "deserialize_bound")]
    pub lb: f64,
    #[serde(serialize_with = "serialize_bound", deserialize_with = "deserialize_bound")]
    pub ub: f64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VariableType {
    Integer,
    Continuous,
    Binary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Constraint {
    pub name: String,
    pub coefficients: Vec<Coefficient>,
    #[serde(serialize_with = "serialize_bound", deserialize_with = "deserialize_bound")]
    pub lhs: f64,
    #[serde(serialize_with = "serialize_bound", deserialize_with = "deserialize_bound")]
    pub rhs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coefficient {
    pub var_name: String,
    pub coeff: f64,
}

// Custom serialization for bounds to handle infinity values
fn serialize_bound<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if value.is_infinite() {
        if value.is_sign_positive() {
            serializer.serialize_str("inf")
        } else {
            serializer.serialize_str("-inf")
        }
    } else {
        serializer.serialize_f64(*value)
    }
}

fn deserialize_bound<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    struct BoundVisitor;

    impl<'de> Visitor<'de> for BoundVisitor {
        type Value = f64;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a number, \"inf\", or \"-inf\"")
        }

        fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(value)
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            match value {
                "inf" => Ok(f64::INFINITY),
                "-inf" => Ok(f64::NEG_INFINITY),
                _ => Err(de::Error::invalid_value(Unexpected::Str(value), &self)),
            }
        }
    }

    deserializer.deserialize_any(BoundVisitor)
}