import { ModelRegistry } from '../modelRegistry';
import { mockModelStore } from '../__mocks__/react-native-nitro-modules';

describe('ModelRegistry Unit Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('isCached should query native store correctly', () => {
    mockModelStore.isCached.mockReturnValueOnce(true);
    const result = ModelRegistry.isCached('https://example.com/test-model.litertlm');
    expect(mockModelStore.isCached).toHaveBeenCalledWith('test-model.litertlm');
    expect(result).toBe(true);
  });

  it('getFilePath should return cached path', () => {
    mockModelStore.getFilePath.mockReturnValueOnce('/caches/test.bin');
    const path = ModelRegistry.getFilePath('test.bin');
    expect(mockModelStore.getFilePath).toHaveBeenCalledWith('test.bin');
    expect(path).toBe('/caches/test.bin');
  });

  it('listCachedFiles should delegate to native', () => {
    const mockFiles = [
      {
        fileName: 'model.bin',
        absolutePath: '/caches/model.bin',
        sizeBytes: 1000,
        lastModifiedMs: 12345,
      },
    ];
    mockModelStore.listCachedFiles.mockReturnValueOnce(mockFiles as any);
    const files = ModelRegistry.listCachedFiles();
    expect(mockModelStore.listCachedFiles).toHaveBeenCalled();
    expect(files).toEqual(mockFiles);
  });

  it('deleteFile should delegate delete to native', () => {
    ModelRegistry.deleteFile('https://example.com/model.bin?q=1');
    expect(mockModelStore.deleteFile).toHaveBeenCalledWith('model.bin');
  });

  it('resolveModel should throw error on HTTP urls', async () => {
    await expect(ModelRegistry.resolveModel('http://example.com/model.bin'))
      .rejects.toThrow('Insecure HTTP URLs are not allowed for model downloads');
  });

  it('resolveModel should download HTTPS urls', async () => {
    mockModelStore.downloadFile.mockResolvedValueOnce('/downloaded/model.bin');
    const path = await ModelRegistry.resolveModel('https://example.com/model.bin', {
      headers: { Authorization: 'Bearer test' },
    });
    expect(mockModelStore.downloadFile).toHaveBeenCalledWith(
      'https://example.com/model.bin',
      'model.bin',
      JSON.stringify({ Authorization: 'Bearer test' }),
      expect.any(Function)
    );
    expect(path).toBe('/downloaded/model.bin');
  });

  it('resolveModel should return local paths directly', async () => {
    const path = await ModelRegistry.resolveModel('/local/path/model.bin');
    expect(mockModelStore.downloadFile).not.toHaveBeenCalled();
    expect(path).toBe('/local/path/model.bin');
  });

  it('resolveModel should strip file:// prefix from local paths', async () => {
    const path = await ModelRegistry.resolveModel('file:///local/path/model.bin');
    expect(mockModelStore.downloadFile).not.toHaveBeenCalled();
    expect(path).toBe('/local/path/model.bin');
  });
});
