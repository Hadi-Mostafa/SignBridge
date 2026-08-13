"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const source = fs.readFileSync("frontend/app.js", "utf8");
const sandbox = {
    window: {},
    document: {
        createElement() {
            return { width: 0, height: 0 };
        },
        readyState: "loading",
        addEventListener() {},
    },
    console,
    performance: { now: () => 0 },
    AbortController,
    URL,
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: "frontend/app.js" });

const helpers = sandbox.window.SignBridgeTest;
assert.ok(helpers, "test helpers are exposed");

const ranked = helpers.normalizeContextSuggestions({
    suggestions: [
        { word: "banana", score: 0.2 },
        { word: "birthday", score: 0.9 },
        { word: "day", score: 1.0 },
        { word: "birthday", score: 0.1 },
    ],
}, "b");
assert.deepEqual([...ranked], ["birthday", "banana"]);
assert.deepEqual([...helpers.sentenceContextWords("Happy birthday! ")], ["happy", "birthday"]);

const committed = helpers.commitWordTransition({ sentence: "Happy ", spelling: "bir" }, "birthday");
assert.equal(committed.sentence, "Happy birthday ");
assert.equal(committed.spelling, "");

const stillCooling = helpers.getLetterAcceptanceTransition({
    label: "A",
    armed: true,
    lastAcceptedLabel: "B",
    lastAcceptedAt: 100,
    now: 600,
});
assert.equal(stillCooling.accepted, false);
assert.equal(stillCooling.reason, "cooldown");

const accepted = helpers.getLetterAcceptanceTransition({
    label: "A",
    armed: true,
    lastAcceptedLabel: "B",
    lastAcceptedAt: 100,
    now: 1_100,
});
assert.equal(accepted.accepted, true);

const landmarks = helpers.normalizeOverlayLandmarks(Array.from({ length: 21 }, (_, index) => [index / 20, 0.5, 0]));
assert.equal(landmarks.length, 21);
assert.deepEqual({ ...landmarks[5] }, { x: 0.25, y: 0.5 });
assert.deepEqual([...helpers.normalizeOverlayLandmarks([[0, 0]])], []);

console.log("frontend state smoke: 13 assertions passed");
