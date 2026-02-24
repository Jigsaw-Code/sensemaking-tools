"use strict";
var __importDefault =
  (this && this.__importDefault) ||
  function (mod) {
    return mod && mod.__esModule ? mod : { default: mod };
  };
Object.defineProperty(exports, "__esModule", { value: true });
const dotenv_1 = __importDefault(require("dotenv"));
const path_1 = __importDefault(require("path"));
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const joi_1 = __importDefault(require("joi"));
const sensemaker_1 = require("../library/src/sensemaker");
const body_parser_1 = __importDefault(require("body-parser"));
const vertex_model_1 = require("../library/src//models/vertex_model");
const types_1 = require("../library/src//types");
dotenv_1.default.config({ path: path_1.default.resolve(__dirname, ".env") }); // Load env vars from backend directory
const app = (0, express_1.default)();
const port = 3000;
// CORS middleware setup
const allowedOrigins = ["http://localhost:4200"];
if (process.env.ALLOWED_ORIGINS) {
  // to allow more origins, add them to `ALLOWED_ORIGINS` env var
  allowedOrigins.push(...process.env.ALLOWED_ORIGINS.split(","));
}
app.use(
  (0, cors_1.default)({
    origin: allowedOrigins,
    credentials: true,
  })
);
// The default body-parser request size limit is 100k, which isn't sufficient for including group vote tally data
app.use(body_parser_1.default.json({ limit: "50mb" }));
const SENSEMAKER = new sensemaker_1.Sensemaker({
  defaultModel: new vertex_model_1.VertexModel(
    process.env.GOOGLE_CLOUD_PROJECT,
    "us-central1",
    "gemini-1.5-pro-002"
  ),
});
const ADDITIONAL_INSTRUCTIONS_SCHEMA = joi_1.default.string().optional();
// Comments array must exist, be non-empty, and contain valid comment data.
const COMMENTS_SCHEMA = joi_1.default
  .array()
  .items(
    joi_1.default
      .object()
      .keys({
        id: joi_1.default.string().required(),
        text: joi_1.default.string().required(),
        voteTalliesByGroup: joi_1.default.object(),
      })
      .required()
  )
  .required();
// TODO: this should be a list of strings representing topic names.
const TOPICS_SCHEMA = joi_1.default.string().optional();
const basicSummarizeSchema = joi_1.default.object().keys({
  additionalinstructions: ADDITIONAL_INSTRUCTIONS_SCHEMA,
  comments: COMMENTS_SCHEMA,
});
const voteTallySummarizeSchema = joi_1.default.object().keys({
  additionalinstructions: ADDITIONAL_INSTRUCTIONS_SCHEMA,
  commentData: COMMENTS_SCHEMA,
});
const topicsSchema = joi_1.default.object().keys({
  comments: COMMENTS_SCHEMA,
  includeSubtopics: joi_1.default.boolean(),
  topics: TOPICS_SCHEMA,
  additionalinstructions: ADDITIONAL_INSTRUCTIONS_SCHEMA,
});
const categorizeSchema = joi_1.default.object().keys({
  comments: COMMENTS_SCHEMA,
  includeSubtopics: joi_1.default.boolean(),
  topics: TOPICS_SCHEMA,
  additionalinstructions: ADDITIONAL_INSTRUCTIONS_SCHEMA,
  // TODO: remove
  groupByTopic: joi_1.default.boolean().optional(),
});
// Stringify the value but ensure a good formatting (ie includes whitespace.)
function stringifyWithFormat(value) {
  return JSON.stringify(value, null, 2);
}
/**
 * Convert the string representation of array of topic names to Topics if possible.
 * @param topicNames a string that is a comma separated list of topic names
 * @returns a list of Topic objects if topicNames exists or undefined.
 */
function getTopics(topicNames) {
  if (!topicNames) {
    return;
  }
  return topicNames.split(",").map((topicName) => {
    return { name: topicName.trim() };
  });
}
app.post("/basicSummarize", (req, res) => {
  const { error, value } = basicSummarizeSchema.validate(req.body);
  if (error) {
    console.error("error: ", error);
    res.sendStatus(400).json({ error: error.details[0].message });
    return;
  }
  SENSEMAKER.summarize(
    value.comments,
    types_1.SummarizationType.AGGREGATE_VOTE,
    undefined,
    value.additionalinstructions
  ).then((summary) => res.send({ text: summary.getText("MARKDOWN") }));
});
app.post("/voteTallySummarize", (req, res) => {
  const { error, value } = voteTallySummarizeSchema.validate(req.body);
  if (error) {
    console.error("error: ", error);
    res.sendStatus(400).json({ error: error.details[0].message });
    return;
  }
  // TODO: here and everywhere convert additionalinstructions to additionalContext
  SENSEMAKER.summarize(
    value.commentData,
    types_1.SummarizationType.AGGREGATE_VOTE,
    undefined,
    value.additionalinstructions
  ).then((summary) => res.send({ text: summary.getText("MARKDOWN") }));
});
app.post("/topics", (req, res) => {
  const { error, value } = topicsSchema.validate(req.body);
  if (error) {
    console.error("error: ", error);
    res.sendStatus(400).json({ error: error.details[0].message });
    return;
  }
  SENSEMAKER.learnTopics(
    value.comments,
    value.includeSubtopics,
    getTopics(value.topics),
    value.additionalinstructions
  ).then((topics) => res.send({ text: stringifyWithFormat(topics) }));
});
app.post("/categorize", (req, res) => {
  const { error, value } = categorizeSchema.validate(req.body);
  if (error) {
    console.error("error: ", error);
    res.sendStatus(400).json({ error: error.details[0].message });
    return;
  }
  SENSEMAKER.categorizeComments(
    value.comments,
    value.includeSubtopics,
    value.topics ? JSON.parse(value.topics) : undefined,
    value.additionalinstructions
  ).then((comments) => res.send({ text: stringifyWithFormat(comments) }));
});
app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
