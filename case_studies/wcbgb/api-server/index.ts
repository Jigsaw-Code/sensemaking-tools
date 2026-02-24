import dotenv from "dotenv";
import path from "path";
import express, { Express, Request, Response } from "express";
import cors from "cors";
import Joi from "joi";
import { Sensemaker } from "../library/src/sensemaker";
import bodyParser from "body-parser";
import { VertexModel } from "../library/src//models/vertex_model";
import { Comment, SummarizationType, Summary, Topic } from "../library/src//types";

dotenv.config({ path: path.resolve(__dirname, ".env") }); // Load env vars from backend directory

const app: Express = express();
const port = 3000;

// CORS middleware setup
const allowedOrigins = ["http://localhost:4200"];
if (process.env.ALLOWED_ORIGINS) {
  // to allow more origins, add them to `ALLOWED_ORIGINS` env var
  allowedOrigins.push(...process.env.ALLOWED_ORIGINS.split(","));
}
app.use(
  cors({
    origin: allowedOrigins,
    credentials: true,
  })
);

// The default body-parser request size limit is 100k, which isn't sufficient for including group vote tally data
app.use(bodyParser.json({ limit: "50mb" }));

const SENSEMAKER = new Sensemaker({
  defaultModel: new VertexModel(
    process.env.GOOGLE_CLOUD_PROJECT,
    "us-central1",
    "gemini-1.5-pro-002"
  ),
});

const ADDITIONAL_INSTRUCTIONS_SCHEMA = Joi.string().optional();

// Comments array must exist, be non-empty, and contain valid comment data.
const COMMENTS_SCHEMA = Joi.array()
  .items(
    Joi.object()
      .keys({
        id: Joi.string().required(),
        text: Joi.string().required(),
        voteTalliesByGroup: Joi.object(),
      })
      .required()
  )
  .required();
// TODO: this should be a list of strings representing topic names.
const TOPICS_SCHEMA = Joi.string().optional();

const basicSummarizeSchema = Joi.object().keys({
  additionalinstructions: ADDITIONAL_INSTRUCTIONS_SCHEMA,
  comments: COMMENTS_SCHEMA,
});

const voteTallySummarizeSchema = Joi.object().keys({
  additionalinstructions: ADDITIONAL_INSTRUCTIONS_SCHEMA,
  commentData: COMMENTS_SCHEMA,
});

const topicsSchema = Joi.object().keys({
  comments: COMMENTS_SCHEMA,
  includeSubtopics: Joi.boolean(),
  topics: TOPICS_SCHEMA,
  additionalinstructions: ADDITIONAL_INSTRUCTIONS_SCHEMA,
});

const categorizeSchema = Joi.object().keys({
  comments: COMMENTS_SCHEMA,
  includeSubtopics: Joi.boolean(),
  topics: TOPICS_SCHEMA,
  additionalinstructions: ADDITIONAL_INSTRUCTIONS_SCHEMA,
  // TODO: remove
  groupByTopic: Joi.boolean().optional(),
});

// Stringify the value but ensure a good formatting (ie includes whitespace.)
function stringifyWithFormat(value: any): string {
  return JSON.stringify(value, null, 2);
}

/**
 * Convert the string representation of array of topic names to Topics if possible.
 * @param topicNames a string that is a comma separated list of topic names
 * @returns a list of Topic objects if topicNames exists or undefined.
 */
function getTopics(topicNames?: string): Topic[] {
  if (!topicNames) {
    return;
  }
  return topicNames.split(",").map((topicName: string): Topic => {
    return { name: topicName.trim() };
  });
}

app.post("/basicSummarize", (req: Request, res: Response) => {
  const { error, value } = basicSummarizeSchema.validate(req.body);

  if (error) {
    console.error("error: ", error);
    res.sendStatus(400).json({ error: error.details[0].message });
    return;
  }
  SENSEMAKER.summarize(
    value.comments,
    SummarizationType.AGGREGATE_VOTE,
    undefined,
    value.additionalinstructions
  ).then((summary: Summary) => res.send({ text: summary.getText("MARKDOWN") }));
});

app.post("/voteTallySummarize", (req: Request, res: Response) => {
  const { error, value } = voteTallySummarizeSchema.validate(req.body);

  if (error) {
    console.error("error: ", error);
    res.sendStatus(400).json({ error: error.details[0].message });
    return;
  }

  // TODO: here and everywhere convert additionalinstructions to additionalContext
  SENSEMAKER.summarize(
    value.commentData,
    SummarizationType.AGGREGATE_VOTE,
    undefined,
    value.additionalinstructions
  ).then((summary: Summary) => res.send({ text: summary.getText("MARKDOWN") }));
});

app.post("/topics", (req: Request, res: Response) => {
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
  ).then((topics: Topic[]) => res.send({ text: stringifyWithFormat(topics) }));
});

app.post("/categorize", (req: Request, res: Response) => {
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
  ).then((comments: Comment[]) => res.send({ text: stringifyWithFormat(comments) }));
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
