#!/usr/bin/env node
import express from "express";
import bodyParser from "body-parser";
import { Client } from "@notionhq/client";

const app = express();
app.use(bodyParser.json());

const notion = new Client({ auth: process.env.NOTION_API_KEY });
const databaseId = process.env.NOTION_DATABASE_ID;

// List all tasks
app.get("/tasks", async (req, res) => {
    try {
        const response = await notion.databases.query({ database_id: databaseId });
        const tasks = response.results.map((p) => p.properties.Name?.title[0]?.plain_text);
        res.json({ tasks });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: "Failed to fetch tasks" });
    }
});

// Add new task
app.post("/tasks", async (req, res) => {
    const { title } = req.body;
    if (!title) return res.status(400).json({ error: "Title required" });

    try {
        const response = await notion.pages.create({
            parent: { database_id: databaseId },
            properties: {
                Name: { title: [{ text: { content: title } }] },
            },
        });
        res.json({ success: true, id: response.id });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: "Failed to add task" });
    }
});

const port = 4000;
app.listen(port, () => console.log(`✅ Notion MCP server running on http://localhost:${port}`));
