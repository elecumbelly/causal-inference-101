import { appendFile, readdir, readFile, writeFile } from 'node:fs/promises';
import { basename, extname, join } from 'node:path';

const outputDirectory = '_build/html';
const productionOrigin = 'https://causal-inference-101.vercel.app';
const temporaryOrigin = /"domain":"http:\/\/localhost:\d+"/g;
const supportedExtensions = new Set(['.html', '.json']);
const layoutControlsFile = 'layout-controls.js';
const layoutControlsMarker = '/* causal-inference-layout-controls */';

async function filesUnder(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const paths = await Promise.all(entries.map(async (entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? filesUnder(path) : [path];
  }));
  return paths.flat();
}

const outputFiles = await filesUnder(outputDirectory);
let replacements = 0;
for (const path of outputFiles) {
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

const entryClientFiles = outputFiles.filter((path) => (
  basename(path).startsWith('entry.client-') && extname(path) === '.js'
));
if (entryClientFiles.length !== 1) {
  throw new Error(`Expected one MyST entry client bundle, found ${entryClientFiles.length}.`);
}

const layoutControls = await readFile(`assets/${layoutControlsFile}`, 'utf8');
await appendFile(entryClientFiles[0], `\n${layoutControlsMarker}\n${layoutControls}\n`);

console.log(`Set ${replacements} page origins to ${productionOrigin}`);
console.log('Bundled desktop layout controls with the MyST client');
