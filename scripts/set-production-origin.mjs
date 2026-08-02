import { readdir, readFile, writeFile } from 'node:fs/promises';
import { extname, join } from 'node:path';

const outputDirectory = '_build/html';
const productionOrigin = 'https://causal-inference-101.vercel.app';
const temporaryOrigin = /"domain":"http:\/\/localhost:\d+"/g;
const supportedExtensions = new Set(['.html', '.json']);

async function filesUnder(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const paths = await Promise.all(entries.map(async (entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? filesUnder(path) : [path];
  }));
  return paths.flat();
}

let replacements = 0;
for (const path of await filesUnder(outputDirectory)) {
  if (!supportedExtensions.has(extname(path))) continue;

  const source = await readFile(path, 'utf8');
  const updated = source.replace(temporaryOrigin, () => {
    replacements += 1;
    return `"domain":"${productionOrigin}"`;
  });
  if (updated !== source) await writeFile(path, updated);
}

if (replacements === 0) {
  throw new Error('No temporary MyST origins found; the export format may have changed.');
}

console.log(`Set ${replacements} page origins to ${productionOrigin}`);
