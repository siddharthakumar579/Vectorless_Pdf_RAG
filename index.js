import { GoogleGenAI } from "@google/genai";
import fs from "fs";
import { extractText } from "unpdf";
import "dotenv/config";


// reads the PDF file and returns raw text
async function extractTextFromPDF(pdfPath) {
  const buffer   = fs.readFileSync(pdfPath);
  const { text } = await extractText(new Uint8Array(buffer));
  return text.join(" ");  // ← join pages into one string
}

// ----------------------------------------------------------------
// STEP 1 — CHUNKING
// Split raw text into overlapping windows of words.
// ----------------------------------------------------------------

function chunkText(rawText, chunkSize = 300, overlap = 50) {
  const words = rawText.split(/\s+/).filter(w => w.length > 0);
  const chunks = [];

  let i = 0;
  while (i < words.length) {
    const chunk = words.slice(i, i + chunkSize).join(" ");
    chunks.push({ id: chunks.length, text: chunk });

    i += chunkSize - overlap; // move forward, leaving overlap behind
    if (i + overlap >= words.length) break;
  }

  // catch any remaining words at the end
  if (i < words.length) {
    chunks.push({ id: chunks.length, text: words.slice(i).join(" ") });
  }

  return chunks;
}


// ----------------------------------------------------------------
// STEP 2 — BM25-STYLE KEYWORD RETRIEVAL 
// For each chunk, count how many times the query's words appear. 
// Higher count = more relevant. No embeddings needed. 
// ----------------------------------------------------------------

function scoreChunk(queryWords, chunkText) {
  const text = chunkText.toLowerCase();
  let score = 0;

  for (const word of queryWords) {
    const matches = text.match(new RegExp(`\\b${word}\\b`, "gi"));
    if (matches) score += matches.length; // count every occurrence
  }

  return score;
}

function retrieveTopChunks(query, chunks, topK = 4) {
  // tokenise query, ignore very short words like "is", "a", "of"
  const queryWords = query
    .toLowerCase()
    .split(/\s+/)
    .filter(w => w.length > 2);

  return chunks
    .map(chunk => ({ ...chunk, score: scoreChunk(queryWords, chunk.text) }))
    .filter(chunk => chunk.score > 0)       // drop chunks with zero matches
    .sort((a, b) => b.score - a.score)      // highest score first
    .slice(0, topK);                         // take top K
}


// ----------------------------------------------------------------
// STEP 3 — BUILD PROMPT
// Pack the retrieved chunks into a prompt.
// LLM Model only sees these chunks, not the full document.
// ----------------------------------------------------------------

function buildPrompt(question, topChunks) {
  const context = topChunks
    .map((c, i) => `[Excerpt ${i + 1}]\n${c.text}`)
    .join("\n\n---\n\n");

  return (
    `Answer the question using ONLY the excerpts below.\n` +
    `If the answer isn't there, say "I couldn't find that in the document."\n\n` +
    `${context}\n\nQuestion: ${question}\nAnswer:`
  );
}


// ----------------------------------------------------------------
// STEP 4 — CALL LLM
// Send the prompt to the LLM API and get an answer back.
// ----------------------------------------------------------------

async function askGemini(prompt, apiKey) {
  const ai     = new GoogleGenAI({ apiKey });
  const result = await ai.models.generateContent({
    model: "gemini-2.5-flash-lite",
    contents: prompt,
  });

  return result.text;
}


// ----------------------------------------------------------------
// PUTTING IT ALL TOGETHER
// ----------------------------------------------------------------

async function vectorlessRAG(rawText, question, apiKey) {
  // 1. split the full document into chunks
  const chunks = chunkText(rawText);

  // 2. find the most relevant chunks using keyword scoring
  const topChunks = retrieveTopChunks(question, chunks);

  // 3. build a prompt with only those chunks
  const prompt = buildPrompt(question, topChunks);

  // 4. ask LLM and return the answer
  const answer = await askGemini(prompt, apiKey);

  return answer;
}


// --------------------------------------------------------------
// USAGE — Add the pdfPath and your desired question
//----------------------------------------------------------------

const rawText = await extractTextFromPDF("./radha.pdf");
const answer  = await vectorlessRAG(rawText, "What is this document about?", process.env.GEMINI_API_KEY);
console.log(answer);
